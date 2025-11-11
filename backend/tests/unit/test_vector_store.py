"""
Unit tests for Vector Store Service

Tests cover:
- Embedding initialization (OpenAI and HuggingFace)
- Document addition (single and batch)
- Similarity search
- Error handling
- Database operations
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.vector_store import VectorStore, VectorStoreError, EmbeddingError


class TestVectorStoreInitialization:
    """Test vector store initialization"""

    @patch('app.services.vector_store.settings')
    @patch('app.services.vector_store.OpenAIEmbeddings')
    def test_initialize_with_openai(self, mock_openai, mock_settings):
        """Test initialization with OpenAI embeddings"""
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_settings.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

        mock_db = Mock()
        mock_db.execute = Mock(return_value=None)
        mock_db.commit = Mock()

        vs = VectorStore(db=mock_db)

        assert vs.dimension == 1536
        mock_openai.assert_called_once()

    @patch('app.services.vector_store.settings')
    @patch('app.services.vector_store.HuggingFaceEmbeddings')
    def test_initialize_with_huggingface(self, mock_hf, mock_settings):
        """Test initialization with HuggingFace embeddings (fallback)"""
        mock_settings.OPENAI_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.GEMINI_MODEL = "gemini-pro"
        mock_settings.GEMINI_EMBEDDING_MODEL = "models/embedding-001"

        mock_db = Mock()
        mock_db.execute = Mock(return_value=None)
        mock_db.commit = Mock()

        vs = VectorStore(db=mock_db)

        assert vs.dimension == 384
        mock_hf.assert_called_once()

    @patch('app.services.vector_store.settings')
    def test_initialization_creates_table(self, mock_settings):
        """Test that initialization creates required tables"""
        mock_settings.OPENAI_API_KEY = "sk-test"

        mock_db = Mock()
        execute_calls = []

        def track_execute(sql):
            execute_calls.append(str(sql))
            return None

        mock_db.execute = Mock(side_effect=track_execute)
        mock_db.commit = Mock()

        with patch('app.services.vector_store.OpenAIEmbeddings'):
            vs = VectorStore(db=mock_db)

        # Verify pgvector extension was enabled
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in str(call)
                  for call in execute_calls)


class TestDocumentAddition:
    """Test adding documents to vector store"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"
            mock_settings.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

            mock_db = Mock()
            mock_db.execute = Mock(return_value=Mock(fetchone=Mock(return_value=[1])))
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings'):
                vs = VectorStore(db=mock_db)

                # Mock embedding generation
                vs._get_embedding_sync = Mock(
                    return_value=np.random.rand(1536).astype(np.float32)
                )

                return vs

    def test_add_document_success(self, mock_vector_store):
        """Test successfully adding a single document"""
        content = "DPI is a key metric for measuring fund performance"
        metadata = {"document_id": 1, "fund_id": 1, "section": "definitions"}

        doc_id = mock_vector_store.add_document_sync(content, metadata)

        assert doc_id == 1
        mock_vector_store._get_embedding_sync.assert_called_once_with(content)
        mock_vector_store.db.execute.assert_called()
        mock_vector_store.db.commit.assert_called()

    def test_add_document_empty_content(self, mock_vector_store):
        """Test that empty content raises ValueError"""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            mock_vector_store.add_document_sync("", {"document_id": 1, "fund_id": 1})

    def test_add_document_missing_metadata(self, mock_vector_store):
        """Test that missing required metadata raises ValueError"""
        content = "Test content"

        # Missing fund_id
        with pytest.raises(ValueError, match="must include"):
            mock_vector_store.add_document_sync(content, {"document_id": 1})

        # Missing document_id
        with pytest.raises(ValueError, match="must include"):
            mock_vector_store.add_document_sync(content, {"fund_id": 1})

    def test_add_documents_batch_success(self, mock_vector_store):
        """Test batch adding multiple documents"""
        documents = [
            {
                "text": "DPI measures distributions to paid-in capital",
                "metadata": {"document_id": 1, "fund_id": 1}
            },
            {
                "text": "IRR is the internal rate of return",
                "metadata": {"document_id": 1, "fund_id": 1}
            },
            {
                "text": "TVPI includes both realized and unrealized value",
                "metadata": {"document_id": 1, "fund_id": 1}
            }
        ]

        # Mock batch embedding generation
        mock_vector_store._get_embeddings_batch = Mock(
            return_value=[np.random.rand(1536).astype(np.float32) for _ in documents]
        )

        count = mock_vector_store.add_documents_batch(documents, batch_size=10)

        assert count == 3
        mock_vector_store._get_embeddings_batch.assert_called_once()

    def test_add_documents_batch_empty_list(self, mock_vector_store):
        """Test that empty document list returns 0"""
        count = mock_vector_store.add_documents_batch([])
        assert count == 0

    def test_add_documents_batch_partial_failure(self, mock_vector_store):
        """Test batch processing with some invalid documents"""
        documents = [
            {
                "text": "Valid document",
                "metadata": {"document_id": 1, "fund_id": 1}
            },
            {
                "text": "Invalid - missing fund_id",
                "metadata": {"document_id": 2}  # Missing fund_id
            },
            {
                "text": "Another valid document",
                "metadata": {"document_id": 3, "fund_id": 1}
            }
        ]

        mock_vector_store._get_embeddings_batch = Mock(
            return_value=[np.random.rand(1536).astype(np.float32) for _ in documents]
        )

        count = mock_vector_store.add_documents_batch(documents, batch_size=10)

        # Should add 2 out of 3 documents
        assert count == 2


class TestSimilaritySearch:
    """Test semantic similarity search"""

    @pytest.fixture
    def mock_vector_store_with_search(self):
        """Create vector store with mocked search results"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            mock_db = Mock()

            # Mock search results
            mock_results = [
                (1, 1, 1, "DPI is a key metric", {"section": "definitions"}, 0.95),
                (2, 1, 1, "Distributions to paid-in capital", {"section": "formulas"}, 0.87),
                (3, 1, 1, "LPs receive distributions", {"section": "general"}, 0.75)
            ]

            mock_db.execute = Mock(return_value=mock_results)
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings'):
                vs = VectorStore(db=mock_db)
                vs._get_embedding_sync = Mock(
                    return_value=np.random.rand(1536).astype(np.float32)
                )
                return vs

    def test_similarity_search_basic(self, mock_vector_store_with_search):
        """Test basic similarity search"""
        results = mock_vector_store_with_search.similarity_search(
            query="What is DPI?",
            k=5
        )

        assert len(results) == 3
        assert results[0]["score"] == 0.95
        assert "DPI" in results[0]["content"]

    def test_similarity_search_with_fund_filter(self, mock_vector_store_with_search):
        """Test similarity search with fund_id filter"""
        results = mock_vector_store_with_search.similarity_search(
            query="What is DPI?",
            k=5,
            filter_metadata={"fund_id": 1}
        )

        assert len(results) == 3
        # Verify fund_id filter was applied in query
        mock_vector_store_with_search.db.execute.assert_called()

    def test_similarity_search_with_threshold(self, mock_vector_store_with_search):
        """Test similarity search with similarity threshold"""
        results = mock_vector_store_with_search.similarity_search(
            query="What is DPI?",
            k=5,
            similarity_threshold=0.8
        )

        # Results should be filtered by threshold
        assert all(r["score"] >= 0.8 for r in results if "score" in r)

    def test_similarity_search_error_handling(self):
        """Test similarity search error handling"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            # Create a mock database that succeeds for initialization but fails for search
            mock_db = Mock()
            call_count = [0]

            def mock_execute_side_effect(sql, params=None):
                call_count[0] += 1
                # First two calls are for initialization (CREATE EXTENSION, CREATE TABLE)
                if call_count[0] <= 2:
                    return None
                # Subsequent calls (search) should fail
                raise Exception("Database error")

            mock_db.execute = Mock(side_effect=mock_execute_side_effect)
            mock_db.commit = Mock()
            mock_db.rollback = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings'):
                vs = VectorStore(db=mock_db)
                vs._get_embedding_sync = Mock(
                    return_value=np.random.rand(1536).astype(np.float32)
                )

                # Should return empty list on error, not raise
                results = vs.similarity_search("test query")
                assert results == []


class TestVectorStoreUtilities:
    """Test utility functions"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            mock_db = Mock()
            mock_db.execute = Mock(return_value=Mock(fetchone=Mock(return_value=[42])))
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings'):
                return VectorStore(db=mock_db)

    def test_get_document_count_all(self, mock_vector_store):
        """Test getting total document count"""
        count = mock_vector_store.get_document_count()
        assert count == 42

    def test_get_document_count_by_fund(self, mock_vector_store):
        """Test getting document count for specific fund"""
        count = mock_vector_store.get_document_count(fund_id=1)
        assert count == 42

    def test_clear_all_documents(self, mock_vector_store):
        """Test clearing all documents"""
        mock_vector_store.clear()
        mock_vector_store.db.execute.assert_called()
        mock_vector_store.db.commit.assert_called()

    def test_clear_by_fund_id(self, mock_vector_store):
        """Test clearing documents for specific fund"""
        mock_vector_store.clear(fund_id=1)
        mock_vector_store.db.execute.assert_called()
        mock_vector_store.db.commit.assert_called()

    def test_clear_by_document_id(self, mock_vector_store):
        """Test clearing specific document"""
        mock_vector_store.clear(document_id=5)
        mock_vector_store.db.execute.assert_called()
        mock_vector_store.db.commit.assert_called()


class TestEmbeddingGeneration:
    """Test embedding generation"""

    def test_embedding_generation_langchain(self):
        """Test embedding generation with LangChain embeddings"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            mock_db = Mock()
            mock_db.execute = Mock()
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings') as mock_openai:
                mock_embeddings = Mock()
                mock_embeddings.embed_query = Mock(return_value=[0.1] * 1536)
                mock_openai.return_value = mock_embeddings

                vs = VectorStore(db=mock_db)
                embedding = vs._get_embedding_sync("test text")

                assert isinstance(embedding, np.ndarray)
                assert embedding.shape == (1536,)
                assert embedding.dtype == np.float32

    def test_batch_embedding_generation(self):
        """Test batch embedding generation"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            mock_db = Mock()
            mock_db.execute = Mock()
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings') as mock_openai:
                mock_embeddings = Mock()
                mock_embeddings.embed_documents = Mock(
                    return_value=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
                )
                mock_openai.return_value = mock_embeddings

                vs = VectorStore(db=mock_db)
                texts = ["text1", "text2", "text3"]
                embeddings = vs._get_embeddings_batch(texts)

                assert len(embeddings) == 3
                assert all(isinstance(emb, np.ndarray) for emb in embeddings)
                assert all(emb.shape == (1536,) for emb in embeddings)

    def test_embedding_error_handling(self):
        """Test embedding generation error handling"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            mock_db = Mock()
            mock_db.execute = Mock()
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings') as mock_openai:
                mock_embeddings = Mock()
                mock_embeddings.embed_query = Mock(side_effect=Exception("API Error"))
                mock_openai.return_value = mock_embeddings

                vs = VectorStore(db=mock_db)

                with pytest.raises(EmbeddingError):
                    vs._get_embedding_sync("test text")


class TestIndexCreation:
    """Test vector index creation"""

    def test_index_creation_sufficient_rows(self):
        """Test index creation when sufficient rows exist"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            mock_db = Mock()

            # Mock row count > 1000
            mock_db.execute = Mock(
                return_value=Mock(fetchone=Mock(return_value=[1500]))
            )
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings'):
                vs = VectorStore(db=mock_db)
                vs._create_vector_index()

                # Should create index
                assert mock_db.commit.called

    def test_index_creation_insufficient_rows(self):
        """Test index creation skipped when insufficient rows"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "sk-test"

            mock_db = Mock()

            # Mock row count < 1000
            mock_db.execute = Mock(
                return_value=Mock(fetchone=Mock(return_value=[500]))
            )
            mock_db.commit = Mock()

            with patch('app.services.vector_store.OpenAIEmbeddings'):
                vs = VectorStore(db=mock_db)

                # Reset mock to track only index creation calls
                mock_db.execute.reset_mock()
                mock_db.commit.reset_mock()

                vs._create_vector_index()

                # Should not create index (only SELECT call)
                assert mock_db.execute.call_count == 1  # Only COUNT query
