"""
Vector Store Service using pgvector (PostgreSQL extension)

This module provides vector storage and semantic search capabilities for document embeddings.
It handles embedding generation, batch processing, and similarity search using PostgreSQL's
pgvector extension.

Key Features:
- Automatic embedding generation (OpenAI or local HuggingFace models)
- Batch processing for efficient large-scale storage
- Semantic similarity search with cosine distance
- Metadata filtering and fund-scoped operations
- Comprehensive error handling and logging
"""
from typing import List, Dict, Any, Optional
import numpy as np
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass


class EmbeddingError(VectorStoreError):
    """Exception raised when embedding generation fails"""
    pass


class VectorStore:
    """
    PostgreSQL-based vector store using pgvector extension

    This class manages document embeddings for semantic search in the RAG pipeline.
    It automatically handles embedding generation, storage, and retrieval.

    Attributes:
        db: SQLAlchemy database session
        embeddings: Embedding model (OpenAI or HuggingFace)
        dimension: Dimension of embedding vectors

    Example:
        >>> vector_store = VectorStore()
        >>> await vector_store.add_document(
        ...     content="DPI is a key metric",
        ...     metadata={"document_id": 1, "fund_id": 1}
        ... )
        >>> results = await vector_store.similarity_search("What is DPI?", k=5)
    """

    def __init__(self, db: Session = None):
        """
        Initialize vector store

        Args:
            db: Optional database session (creates new if not provided)
        """
        self.db = db or SessionLocal()
        self.embeddings = self._initialize_embeddings()
        self.dimension = self._get_embedding_dimension()
        self._ensure_extension()

    def _initialize_embeddings(self):
        """
        Initialize embedding model based on configuration

        Priority:
        1. OpenAI embeddings (if API key is configured)
        2. Google Gemini embeddings (if API key is configured)
        3. Local HuggingFace embeddings (fallback)

        Returns:
            Embedding model instance

        Raises:
            EmbeddingError: If embedding initialization fails
        """
        try:
            if settings.OPENAI_API_KEY:
                logger.info("Initializing OpenAI embeddings (text-embedding-3-small)")
                return OpenAIEmbeddings(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            elif settings.GOOGLE_API_KEY:
                logger.info("Initializing Google Gemini embeddings (embedding-001)")
                return GoogleGenerativeAIEmbeddings(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    google_api_key=settings.GOOGLE_API_KEY
                )
            else:
                logger.info("Initializing local HuggingFace embeddings (all-MiniLM-L6-v2)")
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise EmbeddingError(f"Embedding initialization failed: {e}")

    def _get_embedding_dimension(self) -> int:
        """
        Get dimension of embedding vectors based on model

        Returns:
            Vector dimension (1536 for OpenAI, 768 for Gemini, 384 for HuggingFace)
        """
        if settings.OPENAI_API_KEY:
            return 1536  # OpenAI text-embedding-3-small
        elif settings.GOOGLE_API_KEY:
            return 768   # Google Gemini embedding-001
        else:
            return 384   # sentence-transformers/all-MiniLM-L6-v2

    def _ensure_extension(self):
        """
        Ensure pgvector extension and required tables exist

        Creates:
        - pgvector extension in PostgreSQL
        - document_embeddings table with proper schema
        - Indexes for efficient querying

        Raises:
            VectorStoreError: If setup fails
        """
        try:
            logger.info("Setting up pgvector extension and tables")

            # Enable pgvector extension
            self.db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Create embeddings table with proper constraints
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL,
                fund_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({self.dimension}) NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_fund FOREIGN KEY(fund_id)
                    REFERENCES funds(id) ON DELETE CASCADE
            );

            -- Index on fund_id for faster filtering
            CREATE INDEX IF NOT EXISTS idx_document_embeddings_fund_id
                ON document_embeddings(fund_id);

            -- Index on document_id for lookup
            CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_id
                ON document_embeddings(document_id);
            """

            self.db.execute(text(create_table_sql))
            self.db.commit()

            logger.info("pgvector setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup pgvector: {e}")
            self.db.rollback()
            raise VectorStoreError(f"pgvector setup failed: {e}")

    def _create_vector_index(self):
        """
        Create IVFFLAT index for faster vector similarity search

        Note: Only creates index when there are sufficient rows (>1000)
        for optimal performance. Uses IVFFlat algorithm which partitions
        the vector space for faster approximate nearest neighbor search.
        """
        try:
            # Check row count
            count_result = self.db.execute(
                text("SELECT COUNT(*) FROM document_embeddings")
            ).fetchone()

            row_count = count_result[0] if count_result else 0

            if row_count >= 1000:
                # Drop existing index
                self.db.execute(text(
                    "DROP INDEX IF EXISTS idx_document_embeddings_vector"
                ))

                # Calculate optimal number of lists (rows / 1000 is a good heuristic)
                lists = max(100, row_count // 1000)

                # Create IVFFlat index for cosine distance
                create_index_sql = f"""
                CREATE INDEX idx_document_embeddings_vector
                ON document_embeddings
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
                """

                self.db.execute(text(create_index_sql))
                self.db.commit()
                logger.info(f"Created IVFFlat index with {lists} lists for {row_count} documents")
            else:
                logger.debug(f"Skipping index creation (only {row_count} documents, need 1000+)")

        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
            self.db.rollback()

    def add_document_sync(self, content: str, metadata: Dict[str, Any]) -> int:
        """
        Add a single document to the vector store (synchronous version)

        Args:
            content: Text content to embed and store
            metadata: Document metadata (must include document_id and fund_id)

        Returns:
            ID of the inserted embedding record

        Raises:
            ValueError: If content is empty or metadata is invalid
            VectorStoreError: If insertion fails
        """
        # Validation
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")

        if "document_id" not in metadata or "fund_id" not in metadata:
            raise ValueError("Metadata must include 'document_id' and 'fund_id'")

        try:
            # Generate embedding
            embedding = self._get_embedding_sync(content)
            embedding_list = embedding.tolist()

            # Insert into database
            insert_sql = text("""
                INSERT INTO document_embeddings
                    (document_id, fund_id, content, embedding, metadata)
                VALUES
                    (:document_id, :fund_id, :content, :embedding::vector, :metadata::jsonb)
                RETURNING id
            """)

            result = self.db.execute(insert_sql, {
                "document_id": metadata["document_id"],
                "fund_id": metadata["fund_id"],
                "content": content,
                "embedding": str(embedding_list),
                "metadata": json.dumps(metadata)
            })

            self.db.commit()
            inserted_id = result.fetchone()[0]

            logger.debug(f"Added embedding (id={inserted_id}, fund_id={metadata['fund_id']})")
            return inserted_id

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            self.db.rollback()
            raise VectorStoreError(f"Document insertion failed: {e}")

    def add_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> int:
        """
        Add multiple documents efficiently using batch processing

        This method processes documents in batches to optimize embedding
        generation and database insertion. Recommended for bulk operations.

        Args:
            documents: List of dicts with 'text' and 'metadata' keys
            batch_size: Number of documents to process per batch (default: 50)

        Returns:
            Number of documents successfully added

        Example:
            >>> docs = [
            ...     {"text": "DPI measures distributions", "metadata": {"document_id": 1, "fund_id": 1}},
            ...     {"text": "IRR measures returns", "metadata": {"document_id": 1, "fund_id": 1}}
            ... ]
            >>> count = vector_store.add_documents_batch(docs)
        """
        if not documents:
            logger.warning("add_documents_batch called with empty document list")
            return 0

        added_count = 0
        failed_count = 0
        total_batches = (len(documents) + batch_size - 1) // batch_size

        logger.info(f"Processing {len(documents)} documents in {total_batches} batches")

        for batch_idx in range(0, len(documents), batch_size):
            batch = documents[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            try:
                # Extract texts for batch embedding
                texts = [doc['text'] for doc in batch]

                # Generate embeddings for entire batch
                embeddings = self._get_embeddings_batch(texts)

                # Prepare batch insert values
                values = []
                for i, doc in enumerate(batch):
                    metadata = doc['metadata']

                    # Validate metadata
                    if 'document_id' not in metadata or 'fund_id' not in metadata:
                        logger.warning(f"Skipping document {i} in batch {batch_num}: missing metadata")
                        failed_count += 1
                        continue

                    embedding_list = embeddings[i].tolist()

                    values.append({
                        'document_id': metadata['document_id'],
                        'fund_id': metadata['fund_id'],
                        'content': doc['text'],
                        'embedding': str(embedding_list),
                        'metadata': json.dumps(metadata)
                    })

                # Batch insert into database
                if values:
                    insert_sql = text("""
                        INSERT INTO document_embeddings
                            (document_id, fund_id, content, embedding, metadata)
                        VALUES
                            (:document_id, :fund_id, :content, :embedding::vector, :metadata::jsonb)
                    """)

                    # Execute each insert individually in a transaction
                    for value_dict in values:
                        self.db.execute(insert_sql, value_dict)
                    self.db.commit()

                    added_count += len(values)
                    logger.info(f"Batch {batch_num}/{total_batches}: Added {len(values)} documents")

            except Exception as e:
                logger.error(f"Error in batch {batch_num}/{total_batches}: {e}", exc_info=True)
                self.db.rollback()
                failed_count += len(batch)
                # Re-raise if all batches fail (helps with debugging)
                if batch_num == 1 and failed_count == len(documents):
                    raise
                continue

        # Update vector index after bulk insert
        if added_count > 0:
            self._create_vector_index()

        logger.info(f"Batch processing complete: {added_count} added, {failed_count} failed")
        return added_count

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar documents using cosine similarity

        This method generates an embedding for the query and finds the k most
        similar documents in the vector store using pgvector's cosine distance operator.

        Args:
            query: Search query text
            k: Number of results to return (default: 5)
            filter_metadata: Optional filters (e.g., {"fund_id": 1})
            similarity_threshold: Minimum similarity score 0-1 (optional)

        Returns:
            List of documents sorted by similarity score (highest first)
            Each document contains: id, content, metadata, score

        Example:
            >>> results = vector_store.similarity_search(
            ...     query="What is DPI?",
            ...     k=5,
            ...     filter_metadata={"fund_id": 1},
            ...     similarity_threshold=0.7
            ... )
        """
        try:
            # Generate query embedding
            query_embedding = self._get_embedding_sync(query)
            embedding_list = query_embedding.tolist()

            # Build WHERE clause for filters
            where_clauses = []
            params = {
                "query_embedding": str(embedding_list),
                "k": k
            }

            # Add metadata filters
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key in ["document_id", "fund_id"]:
                        where_clauses.append(f"{key} = :{key}")
                        params[key] = value

            # Add similarity threshold filter
            if similarity_threshold is not None:
                where_clauses.append(
                    "(1 - (embedding <=> :query_embedding::vector)) >= :threshold"
                )
                params["threshold"] = similarity_threshold

            where_clause = ""
            if where_clauses:
                where_clause = "WHERE " + " AND ".join(where_clauses)

            # Execute similarity search using pgvector cosine distance operator (<=>)
            # Lower distance = higher similarity, so we convert to similarity score: 1 - distance
            search_sql = text(f"""
                SELECT
                    id,
                    document_id,
                    fund_id,
                    content,
                    metadata,
                    1 - (embedding <=> :query_embedding::vector) as similarity_score
                FROM document_embeddings
                {where_clause}
                ORDER BY embedding <=> :query_embedding::vector
                LIMIT :k
            """)

            result = self.db.execute(search_sql, params)

            # Format results
            results = []
            for row in result:
                score = float(row[5])
                # Apply post-filtering for similarity threshold (defensive check)
                if similarity_threshold is not None and score < similarity_threshold:
                    continue
                results.append({
                    "id": row[0],
                    "document_id": row[1],
                    "fund_id": row[2],
                    "content": row[3],
                    "metadata": row[4],
                    "score": score
                })

            logger.debug(f"Similarity search for '{query[:50]}...' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def _get_embedding_sync(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text (synchronous)

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            if hasattr(self.embeddings, 'embed_query'):
                # LangChain embeddings
                embedding = self.embeddings.embed_query(text)
            else:
                # HuggingFace embeddings
                embedding = self.embeddings.encode(text)

            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batch processing)

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        try:
            if hasattr(self.embeddings, 'embed_documents'):
                # LangChain embeddings (supports batch)
                embeddings = self.embeddings.embed_documents(texts)
            else:
                # HuggingFace embeddings (supports batch)
                embeddings = self.embeddings.encode(texts)

            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")

    def get_document_count(self, fund_id: Optional[int] = None) -> int:
        """
        Get count of documents in vector store

        Args:
            fund_id: Optional fund ID to filter by

        Returns:
            Number of documents in vector store
        """
        try:
            if fund_id:
                sql = text("SELECT COUNT(*) FROM document_embeddings WHERE fund_id = :fund_id")
                result = self.db.execute(sql, {"fund_id": fund_id})
            else:
                sql = text("SELECT COUNT(*) FROM document_embeddings")
                result = self.db.execute(sql)

            count = result.fetchone()[0]
            return count

        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def clear(self, fund_id: Optional[int] = None, document_id: Optional[int] = None):
        """
        Clear embeddings from vector store

        Args:
            fund_id: Optional fund ID to filter deletion
            document_id: Optional document ID to filter deletion

        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            if document_id:
                delete_sql = text("DELETE FROM document_embeddings WHERE document_id = :document_id")
                self.db.execute(delete_sql, {"document_id": document_id})
                logger.info(f"Cleared embeddings for document {document_id}")

            elif fund_id:
                delete_sql = text("DELETE FROM document_embeddings WHERE fund_id = :fund_id")
                self.db.execute(delete_sql, {"fund_id": fund_id})
                logger.info(f"Cleared embeddings for fund {fund_id}")

            else:
                delete_sql = text("TRUNCATE TABLE document_embeddings")
                self.db.execute(delete_sql)
                logger.info("Cleared all embeddings from vector store")

            self.db.commit()

        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            self.db.rollback()
            raise VectorStoreError(f"Vector store clear operation failed: {e}")
