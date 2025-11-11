"""
Document processing service using pdfplumber

Extracts tables and text from PDF documents, classifies tables,
and stores structured data in the database.
"""
from typing import Dict, List, Any, Optional
import pdfplumber
import logging
import re
import os
from pathlib import Path
from sqlalchemy.orm import Session
from app.core.config import settings
from app.services.table_parser import TableParser
from app.services.vector_store import VectorStore
from app.db.session import SessionLocal
from app.models.transaction import CapitalCall, Distribution, Adjustment
from app.models.fund import Fund
from app.exceptions import (
    PDFCorruptedError,
    PDFPasswordProtectedError,
    PDFEmptyError,
    PDFNoTablesError,
    InvalidPDFFormatError,
    DocumentProcessingError
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process PDF documents and extract structured data

    This class handles:
    - PDF validation and parsing
    - Table extraction and classification
    - Text chunking for RAG
    - Automatic storage to vector database
    """

    def __init__(self, db=None):
        self.table_parser = TableParser()
        self.vector_store = None  # Initialize lazily to avoid circular dependencies
        self.db = db  # Database session for vector store

    def _validate_pdf(self, file_path: str) -> None:
        """
        Validate PDF file before processing

        Args:
            file_path: Path to PDF file

        Raises:
            FileNotFoundError: If file doesn't exist
            InvalidPDFFormatError: If file is not a PDF or has wrong extension
            PDFCorruptedError: If PDF is corrupted/unreadable
            PDFPasswordProtectedError: If PDF is password protected
            PDFEmptyError: If PDF has no pages or no content
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Check file extension
        file_extension = Path(file_path).suffix.lower()
        if file_extension != '.pdf':
            raise InvalidPDFFormatError(
                f"Invalid file format: {file_extension}. Expected .pdf"
            )

        # Check file size (empty file check)
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise PDFEmptyError("PDF file is empty (0 bytes)")

        # Check minimum file size (typical PDF header is ~10 bytes minimum)
        if file_size < 100:
            raise InvalidPDFFormatError(
                f"File too small to be a valid PDF ({file_size} bytes)"
            )

        # Try to open PDF with pdfplumber to check for corruption and password
        try:
            with pdfplumber.open(file_path) as pdf:
                # Check if PDF has pages
                if not pdf.pages or len(pdf.pages) == 0:
                    raise PDFEmptyError("PDF has no pages")

                # Try to access first page to detect corruption
                first_page = pdf.pages[0]

                # Try to extract text from first page
                try:
                    _ = first_page.extract_text()
                except Exception as e:
                    logger.warning(f"Could not extract text from first page: {e}")
                    # This is not fatal - PDF might have images only

        except pdfplumber.pdfminer.pdfdocument.PDFPasswordIncorrect:
            raise PDFPasswordProtectedError("PDF is password protected")
        except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
            raise PDFCorruptedError(f"PDF syntax error (corrupted file): {str(e)}")
        except pdfplumber.pdfminer.psparser.PSException as e:
            raise PDFCorruptedError(f"PDF parsing error (corrupted file): {str(e)}")
        except Exception as e:
            # Catch other pdfplumber/pdfminer errors
            if "password" in str(e).lower():
                raise PDFPasswordProtectedError(f"PDF is password protected: {str(e)}")
            elif "corrupt" in str(e).lower() or "invalid" in str(e).lower():
                raise PDFCorruptedError(f"PDF appears to be corrupted: {str(e)}")
            else:
                # Re-raise as DocumentProcessingError for unexpected errors
                raise DocumentProcessingError(f"Error validating PDF: {str(e)}")

    async def process_document(self, file_path: str, document_id: int, fund_id: int) -> Dict[str, Any]:
        """
        Process a PDF document - extract tables and text

        Args:
            file_path: Path to the PDF file
            document_id: Database document ID
            fund_id: Fund ID

        Returns:
            Processing result with statistics

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            InvalidPDFFormatError: If file is not a valid PDF
            PDFCorruptedError: If PDF is corrupted
            PDFPasswordProtectedError: If PDF is password protected
            PDFEmptyError: If PDF has no content
        """
        try:
            logger.info(f"Starting document processing: {file_path}")

            # Validate PDF file before processing
            try:
                self._validate_pdf(file_path)
            except (FileNotFoundError, InvalidPDFFormatError, PDFCorruptedError,
                    PDFPasswordProtectedError, PDFEmptyError) as e:
                logger.error(f"PDF validation failed: {str(e)}")
                return {
                    'status': 'failed',
                    'error': str(e),
                    'error_type': type(e).__name__
                }

            # Initialize counters
            stats = {
                'capital_calls': 0,
                'distributions': 0,
                'adjustments': 0,
                'text_chunks': 0,
                'pages_processed': 0,
                'tables_found': 0,
                'tables_classified': 0
            }

            warnings = []

            # Open PDF with pdfplumber
            with pdfplumber.open(file_path) as pdf:
                stats['pages_processed'] = len(pdf.pages)
                logger.info(f"PDF has {len(pdf.pages)} pages")

                all_text = []

                # Process each page
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.debug(f"Processing page {page_num}/{len(pdf.pages)}")

                    # Extract text from page
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)

                    # Extract tables from page
                    tables = page.extract_tables()

                    if tables:
                        logger.info(f"Found {len(tables)} tables on page {page_num}")
                        stats['tables_found'] += len(tables)

                        # Get surrounding context for better classification
                        context = page_text[:500] if page_text else ""

                        # Process each table
                        for table_idx, table in enumerate(tables):
                            if not table or len(table) < 2:
                                continue

                            # Classify table
                            table_type = self.table_parser.classify_table(table, context)

                            if table_type:
                                stats['tables_classified'] += 1
                                logger.info(f"Page {page_num}, Table {table_idx + 1}: Classified as {table_type}")

                                # Parse and store table data
                                saved_count = self._parse_and_store_table(
                                    table,
                                    table_type,
                                    fund_id
                                )

                                if saved_count > 0:
                                    stats[f"{table_type}s"] += saved_count
                            else:
                                logger.debug(f"Page {page_num}, Table {table_idx + 1}: Could not classify")

                # Check if any tables were found (graceful degradation)
                if stats['tables_found'] == 0:
                    warning_msg = "No tables found in PDF. Document may contain images or non-standard table formats."
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)

                # Check if any tables were classified
                if stats['tables_found'] > 0 and stats['tables_classified'] == 0:
                    warning_msg = f"Found {stats['tables_found']} tables but could not classify any. Tables may not match expected formats."
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)

                # Check if we extracted any records
                total_records = (
                    stats['capital_calls'] +
                    stats['distributions'] +
                    stats['adjustments']
                )
                if stats['tables_classified'] > 0 and total_records == 0:
                    warning_msg = f"Classified {stats['tables_classified']} tables but extracted 0 records. Data may be incomplete or malformed."
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)

                # Extract and chunk text for vector storage
                full_text = "\n\n".join(all_text)

                # Check if we have any text content
                if not full_text or len(full_text.strip()) < 50:
                    warning_msg = "PDF has very little or no extractable text. Document may contain only images."
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)

                text_chunks = self._chunk_text(full_text, document_id, fund_id)
                stats['text_chunks'] = len(text_chunks)

                # Store text chunks in vector database for RAG
                if text_chunks:
                    try:
                        stored_count = self._store_chunks_to_vector_db(text_chunks)
                        stats['chunks_stored_in_vector_db'] = stored_count
                        logger.info(f"Stored {stored_count} text chunks to vector database")
                    except Exception as e:
                        warning_msg = f"Failed to store chunks to vector database: {str(e)}"
                        warnings.append(warning_msg)
                        logger.warning(warning_msg)
                        stats['chunks_stored_in_vector_db'] = 0

                # Determine final status
                status = 'completed'
                if warnings:
                    status = 'completed_with_warnings'

                logger.info(f"Document processing completed: {stats}, warnings: {len(warnings)}")

                result = {
                    'status': status,
                    'statistics': stats,
                    'text_chunks': text_chunks  # Will be used for vector storage
                }

                if warnings:
                    result['warnings'] = warnings

                return result

        except DocumentProcessingError as e:
            # Custom exceptions - already logged in validation
            logger.error(f"Document processing error: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error processing document: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': f"Unexpected error: {str(e)}",
                'error_type': 'UnexpectedError'
            }

    def _parse_and_store_table(
        self,
        table_data: List[List[str]],
        table_type: str,
        fund_id: int
    ) -> int:
        """
        Parse table and store records in database

        Args:
            table_data: 2D list of table cells
            table_type: Type of table (capital_call, distribution, adjustment)
            fund_id: Fund ID

        Returns:
            Number of records saved
        """
        db = SessionLocal()
        saved_count = 0

        try:
            # Parse table based on type
            if table_type == 'capital_call':
                records = self.table_parser.parse_capital_call_table(table_data)

                for record in records:
                    capital_call = CapitalCall(
                        fund_id=fund_id,
                        call_date=record['call_date'],
                        call_type=record['call_type'],
                        amount=record['amount'],
                        description=record['description']
                    )
                    db.add(capital_call)
                    saved_count += 1

            elif table_type == 'distribution':
                records = self.table_parser.parse_distribution_table(table_data)

                for record in records:
                    distribution = Distribution(
                        fund_id=fund_id,
                        distribution_date=record['distribution_date'],
                        distribution_type=record['distribution_type'],
                        is_recallable=record['is_recallable'],
                        amount=record['amount'],
                        description=record['description']
                    )
                    db.add(distribution)
                    saved_count += 1

            elif table_type == 'adjustment':
                records = self.table_parser.parse_adjustment_table(table_data)

                for record in records:
                    adjustment = Adjustment(
                        fund_id=fund_id,
                        adjustment_date=record['adjustment_date'],
                        adjustment_type=record['adjustment_type'],
                        category=record['category'],
                        amount=record['amount'],
                        is_contribution_adjustment=record['is_contribution_adjustment'],
                        description=record['description']
                    )
                    db.add(adjustment)
                    saved_count += 1

            # Commit all records
            db.commit()
            logger.info(f"Saved {saved_count} {table_type} records to database")

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving {table_type} records: {str(e)}")
            saved_count = 0
        finally:
            db.close()

        return saved_count

    def _chunk_text(self, text: str, document_id: int, fund_id: int) -> List[Dict[str, Any]]:
        """
        Chunk text content for vector storage

        Uses simple sliding window approach with overlap to preserve context.

        Args:
            text: Full text content from PDF
            document_id: Document ID for metadata
            fund_id: Fund ID for metadata

        Returns:
            List of text chunks with metadata
        """
        if not text or len(text.strip()) == 0:
            return []

        chunks = []
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP

        # Extract special sections first (definitions, strategy, etc.)
        sections = self.table_parser.extract_text_sections(text)

        # Add special sections as separate chunks
        for section_name, section_content in sections.items():
            if len(section_content.strip()) > 50:
                chunks.append({
                    'text': section_content.strip(),
                    'metadata': {
                        'document_id': document_id,
                        'fund_id': fund_id,
                        'section': section_name,
                        'chunk_type': 'section'
                    }
                })

        # Clean text
        text = self._clean_text(text)

        # Split into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'document_id': document_id,
                        'fund_id': fund_id,
                        'chunk_index': chunk_index,
                        'chunk_type': 'general'
                    }
                })
                chunk_index += 1

                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) > 50:  # Minimum chunk length
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'document_id': document_id,
                        'fund_id': fund_id,
                        'chunk_index': chunk_index,
                        'chunk_type': 'general'
                    }
                })

        logger.info(f"Created {len(chunks)} text chunks (including {len(sections)} special sections)")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers/footers patterns
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*\n', '', text)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _store_chunks_to_vector_db(self, text_chunks: List[Dict[str, Any]]) -> int:
        """
        Store text chunks to vector database for RAG

        Args:
            text_chunks: List of text chunks with metadata

        Returns:
            Number of chunks successfully stored

        Raises:
            Exception: If vector store initialization or storage fails
        """
        try:
            # Initialize vector store lazily
            if self.vector_store is None:
                self.vector_store = VectorStore(db=self.db) if self.db else VectorStore()

            # Prepare documents for batch insertion
            documents = []
            for chunk in text_chunks:
                documents.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata']
                })

            # Store in batch for efficiency
            stored_count = self.vector_store.add_documents_batch(documents, batch_size=50)

            return stored_count

        except Exception as e:
            logger.error(f"Failed to store chunks to vector database: {e}")
            raise
