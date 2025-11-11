"""
Unit tests for DocumentProcessor

Tests PDF processing, text chunking, and data storage logic.
Aligned with Evaluation Criteria: Code Quality & Parsing Accuracy
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.document_processor import DocumentProcessor
from app.core.config import settings


class TestTextChunking:
    """Test suite for text chunking functionality"""

    def setup_method(self):
        """Setup test instance"""
        self.processor = DocumentProcessor()

    def test_chunk_text_basic(self, sample_text_for_chunking):
        """Test basic text chunking"""
        chunks = self.processor._chunk_text(sample_text_for_chunking, document_id=1, fund_id=1)

        assert len(chunks) > 0, "Should create at least one chunk"
        assert all('text' in chunk for chunk in chunks), "All chunks should have 'text' field"
        assert all('metadata' in chunk for chunk in chunks), "All chunks should have 'metadata' field"

    def test_chunk_metadata(self, sample_text_for_chunking):
        """Test that chunks have proper metadata"""
        chunks = self.processor._chunk_text(sample_text_for_chunking, document_id=5, fund_id=3)

        for chunk in chunks:
            metadata = chunk['metadata']
            assert 'document_id' in metadata, "Should have document_id in metadata"
            assert 'fund_id' in metadata, "Should have fund_id in metadata"
            assert metadata['document_id'] == 5, "Should have correct document_id"
            assert metadata['fund_id'] == 3, "Should have correct fund_id"

    def test_special_sections_extracted(self, sample_text_for_chunking):
        """Test that special sections are extracted separately"""
        chunks = self.processor._chunk_text(sample_text_for_chunking, document_id=1, fund_id=1)

        section_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'section']
        assert len(section_chunks) > 0, "Should extract special sections"

        section_names = [c['metadata'].get('section') for c in section_chunks]
        assert 'strategy' in section_names, "Should extract strategy section"
        assert 'definitions' in section_names, "Should extract definitions section"

    def test_chunk_text_empty_string(self):
        """Test chunking empty string returns empty list"""
        chunks = self.processor._chunk_text("", document_id=1, fund_id=1)
        assert chunks == [], "Should return empty list for empty text"

    def test_chunk_text_respects_size_limits(self):
        """Test that chunks respect configured size limits"""
        # Create long text
        long_text = "This is a test sentence. " * 200

        chunks = self.processor._chunk_text(long_text, document_id=1, fund_id=1)

        # Check that chunks are within reasonable size
        for chunk in chunks:
            chunk_length = len(chunk['text'])
            # Allow some tolerance for overlap and sentence boundaries
            assert chunk_length <= settings.CHUNK_SIZE * 1.5, "Chunk should not exceed size limit significantly"

    def test_chunk_overlap_preserves_context(self):
        """Test that overlapping chunks preserve context"""
        # Text with clear sentence boundaries
        text = "First sentence. " * 50 + "Second sentence. " * 50

        chunks = self.processor._chunk_text(text, document_id=1, fund_id=1)

        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i]['text'][-100:]  # Last 100 chars
                chunk2_start = chunks[i + 1]['text'][:100]  # First 100 chars

                # There should be some common words (not exact match due to sentence boundaries)
                words1 = set(chunk1_end.split())
                words2 = set(chunk2_start.split())
                common_words = words1.intersection(words2)

                # Should have at least some overlap
                assert len(common_words) > 0, "Consecutive chunks should have word overlap"


class TestTextCleaning:
    """Test suite for text cleaning"""

    def setup_method(self):
        """Setup test instance"""
        self.processor = DocumentProcessor()

    def test_clean_text_removes_extra_whitespace(self):
        """Test that excessive whitespace is removed"""
        text = "This  has   extra    spaces"
        result = self.processor._clean_text(text)
        assert "  " not in result, "Should remove extra spaces"

    def test_clean_text_removes_page_numbers(self):
        """Test that page numbers are removed"""
        text = "Some content Page 5 of 10 more content"
        result = self.processor._clean_text(text)
        assert "Page 5 of 10" not in result, "Should remove page numbers"

    def test_clean_text_removes_excessive_newlines(self):
        """Test that excessive newlines are removed"""
        text = "Line 1\n\n\n\n\nLine 2"
        result = self.processor._clean_text(text)
        assert "\n\n\n" not in result, "Should remove excessive newlines"

    def test_clean_text_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed"""
        text = "  \n  content  \n  "
        result = self.processor._clean_text(text)
        assert result == result.strip(), "Should strip leading/trailing whitespace"


class TestDocumentProcessorIntegration:
    """Integration tests for document processor with mocked PDF"""

    def setup_method(self):
        """Setup test instance"""
        self.processor = DocumentProcessor()

    @pytest.mark.asyncio
    async def test_process_document_file_not_found(self):
        """Test processing non-existent file"""
        result = await self.processor.process_document(
            file_path="/nonexistent/file.pdf",
            document_id=1,
            fund_id=1
        )

        assert result['status'] == 'failed', "Should fail for non-existent file"
        assert 'error' in result, "Should have error message"

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentProcessor._validate_pdf')
    @patch('pdfplumber.open')
    async def test_process_document_success(self, mock_pdf_open, mock_validate):
        """Test successful document processing"""
        # Skip validation
        mock_validate.return_value = None

        # Mock PDF object
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample fund performance report"
        mock_page.extract_tables.return_value = []

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdf_open.return_value = mock_pdf

        result = await self.processor.process_document(
            file_path="/fake/path.pdf",
            document_id=1,
            fund_id=1
        )

        assert result['status'] in ['completed', 'completed_with_warnings'], "Should complete successfully"
        assert 'statistics' in result, "Should have statistics"
        assert result['statistics']['pages_processed'] == 1, "Should process 1 page"

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentProcessor._validate_pdf')
    @patch('pdfplumber.open')
    async def test_process_document_with_tables(self, mock_pdf_open, mock_validate):
        """Test document processing with tables"""
        # Skip validation
        mock_validate.return_value = None

        # Mock capital call table
        capital_call_table = [
            ['Date', 'Call Number', 'Amount', 'Description'],
            ['2023-01-15', 'Call 1', '$5,000,000', 'Initial Capital'],
        ]

        mock_page = Mock()
        mock_page.extract_text.return_value = "Capital Calls section"
        mock_page.extract_tables.return_value = [capital_call_table]

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdf_open.return_value = mock_pdf

        with patch.object(self.processor, '_parse_and_store_table', return_value=1) as mock_store:
            result = await self.processor.process_document(
                file_path="/fake/path.pdf",
                document_id=1,
                fund_id=1
            )

            assert result['status'] in ['completed', 'completed_with_warnings'], "Should complete successfully"
            assert result['statistics']['tables_found'] == 1, "Should find 1 table"
            mock_store.assert_called_once()  # Should attempt to store table

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentProcessor._validate_pdf')
    @patch('pdfplumber.open')
    async def test_process_document_creates_text_chunks(self, mock_pdf_open, mock_validate):
        """Test that document processing creates text chunks"""
        # Skip validation
        mock_validate.return_value = None

        mock_page = Mock()
        mock_page.extract_text.return_value = "Fund Strategy: Focus on technology. " * 50
        mock_page.extract_tables.return_value = []

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=False)

        mock_pdf_open.return_value = mock_pdf

        result = await self.processor.process_document(
            file_path="/fake/path.pdf",
            document_id=1,
            fund_id=1
        )

        assert result['status'] in ['completed', 'completed_with_warnings'], "Should complete successfully"
        assert 'text_chunks' in result, "Should have text_chunks"
        assert len(result['text_chunks']) > 0, "Should create at least one text chunk"
        assert result['statistics']['text_chunks'] > 0, "Should count text chunks"

    @pytest.mark.asyncio
    @patch('app.services.document_processor.DocumentProcessor._validate_pdf')
    @patch('pdfplumber.open')
    async def test_process_document_handles_exceptions(self, mock_pdf_open, mock_validate):
        """Test that document processing handles exceptions gracefully"""
        # Skip validation
        mock_validate.return_value = None

        # Mock pdfplumber to raise exception during processing
        mock_pdf_open.side_effect = Exception("PDF parsing error")

        result = await self.processor.process_document(
            file_path="/fake/path.pdf",
            document_id=1,
            fund_id=1
        )

        assert result['status'] == 'failed', "Should fail gracefully"
        assert 'error' in result, "Should have error message"
        assert 'PDF parsing error' in result['error'], "Should include error details"


class TestParseAndStoreTable:
    """Test suite for table parsing and storage"""

    def setup_method(self):
        """Setup test instance"""
        self.processor = DocumentProcessor()

    @patch('app.services.document_processor.SessionLocal')
    def test_parse_and_store_capital_call(self, mock_session_local, sample_capital_call_table):
        """Test parsing and storing capital call table"""
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        count = self.processor._parse_and_store_table(
            table_data=sample_capital_call_table,
            table_type='capital_call',
            fund_id=1
        )

        assert count == 4, "Should store 4 capital call records"
        assert mock_db.add.call_count == 4, "Should add 4 records to session"
        mock_db.commit.assert_called_once()

    @patch('app.services.document_processor.SessionLocal')
    def test_parse_and_store_distribution(self, mock_session_local, sample_distribution_table):
        """Test parsing and storing distribution table"""
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        count = self.processor._parse_and_store_table(
            table_data=sample_distribution_table,
            table_type='distribution',
            fund_id=1
        )

        assert count == 4, "Should store 4 distribution records"
        assert mock_db.add.call_count == 4, "Should add 4 records to session"
        mock_db.commit.assert_called_once()

    @patch('app.services.document_processor.SessionLocal')
    def test_parse_and_store_adjustment(self, mock_session_local, sample_adjustment_table):
        """Test parsing and storing adjustment table"""
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        count = self.processor._parse_and_store_table(
            table_data=sample_adjustment_table,
            table_type='adjustment',
            fund_id=1
        )

        assert count == 3, "Should store 3 adjustment records"
        assert mock_db.add.call_count == 3, "Should add 3 records to session"
        mock_db.commit.assert_called_once()

    @patch('app.services.document_processor.SessionLocal')
    def test_parse_and_store_handles_errors(self, mock_session_local, sample_capital_call_table):
        """Test that storage errors are handled gracefully"""
        mock_db = Mock()
        mock_db.commit.side_effect = Exception("Database error")
        mock_session_local.return_value = mock_db

        count = self.processor._parse_and_store_table(
            table_data=sample_capital_call_table,
            table_type='capital_call',
            fund_id=1
        )

        assert count == 0, "Should return 0 on error"
        mock_db.rollback.assert_called_once()

    @patch('app.services.document_processor.SessionLocal')
    def test_parse_and_store_unknown_type(self, mock_session_local):
        """Test parsing unknown table type"""
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        count = self.processor._parse_and_store_table(
            table_data=[['header'], ['data']],
            table_type='unknown_type',
            fund_id=1
        )

        # Should not add any records for unknown type
        assert mock_db.add.call_count == 0, "Should not add records for unknown type"
