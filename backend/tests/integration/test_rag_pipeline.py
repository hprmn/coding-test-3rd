"""
Integration tests for RAG Pipeline

Tests the complete end-to-end flow:
1. Document upload and processing
2. Text chunking and embedding storage
3. Semantic search
4. Query processing with RAG
5. Response generation

Note: These tests require a running PostgreSQL database with pgvector extension.
Mark as integration tests to skip in unit test runs.
"""

import os
from pathlib import Path

import pytest
from app.db.session import SessionLocal
from app.services.document_processor import DocumentProcessor
from app.services.query_engine import QueryEngine
from app.services.vector_store import VectorStore


@pytest.mark.integration
@pytest.mark.skipif(not Path("/tmp/test.pdf").exists(), reason="Requires test PDF file")
class TestRAGPipelineIntegration:
    """Integration tests for complete RAG pipeline"""

    @pytest.fixture(scope="class")
    def db_session(self):
        """Get database session for testing"""
        db = SessionLocal()
        yield db
        db.close()

    @pytest.fixture(scope="class")
    def test_fund(self, db_session):
        """Create a test fund"""
        from app.models import Fund

        fund = Fund(
            name="Integration Test Fund",
            gp_name="Test GP",
            fund_type="Venture Capital",
            vintage_year=2024,
        )
        db_session.add(fund)
        db_session.commit()
        db_session.refresh(fund)

        yield fund

        # Cleanup
        db_session.query(Fund).filter(Fund.id == fund.id).delete()
        db_session.commit()

    @pytest.fixture(scope="class")
    def sample_document_chunks(self):
        """Sample document chunks for testing"""
        return [
            {
                "text": "DPI (Distributions to Paid-In Capital) is a key performance metric used in private equity. It measures the ratio of total distributions received by Limited Partners to the total capital they have paid into the fund. A DPI of 1.0 means investors have received back their original investment.",
                "metadata": {
                    "document_id": 1,
                    "fund_id": 1,
                    "section": "definitions",
                    "chunk_type": "section",
                },
            },
            {
                "text": "The formula for calculating DPI is: DPI = Cumulative Distributions / Paid-In Capital. For example, if a fund has called $100M in capital and distributed $40M to LPs, the DPI would be 0.40 or 40%.",
                "metadata": {
                    "document_id": 1,
                    "fund_id": 1,
                    "section": "formulas",
                    "chunk_type": "section",
                },
            },
            {
                "text": "IRR (Internal Rate of Return) is the annualized effective compounded return rate. It accounts for the timing of capital calls and distributions. A higher IRR indicates better fund performance when considering the time value of money.",
                "metadata": {
                    "document_id": 1,
                    "fund_id": 1,
                    "section": "definitions",
                    "chunk_type": "section",
                },
            },
            {
                "text": "Paid-In Capital (PIC) represents the total amount of capital that has been called from Limited Partners. It includes all capital calls minus any adjustments such as returned capital or fees.",
                "metadata": {
                    "document_id": 1,
                    "fund_id": 1,
                    "section": "definitions",
                    "chunk_type": "section",
                },
            },
            {
                "text": "Recallable distributions are distributions that can be called back by the General Partner under certain circumstances. These are typically subtracted from total distributions when calculating net DPI.",
                "metadata": {
                    "document_id": 1,
                    "fund_id": 1,
                    "section": "definitions",
                    "chunk_type": "section",
                },
            },
        ]

    def test_01_vector_store_initialization(self, db_session):
        """Test vector store can be initialized"""
        vector_store = VectorStore(db=db_session)

        assert vector_store is not None
        assert vector_store.dimension in [384, 1536]  # HuggingFace or OpenAI

    def test_02_add_documents_to_vector_store(self, db_session, sample_document_chunks):
        """Test adding documents to vector store"""
        vector_store = VectorStore(db=db_session)

        # Add documents
        count = vector_store.add_documents_batch(sample_document_chunks, batch_size=10)

        assert count == len(sample_document_chunks)

        # Verify documents were added
        total_count = vector_store.get_document_count()
        assert total_count >= count

    def test_03_similarity_search_basic(self, db_session):
        """Test basic similarity search"""
        vector_store = VectorStore(db=db_session)

        # Search for DPI definition
        results = vector_store.similarity_search(query="What is DPI?", k=3)

        assert len(results) > 0
        assert results[0]["score"] > 0.5  # Should have reasonable similarity

        # Check that relevant content is returned
        top_result = results[0]["content"]
        assert "DPI" in top_result or "Distributions" in top_result

    def test_04_similarity_search_with_fund_filter(self, db_session, test_fund):
        """Test similarity search with fund_id filter"""
        vector_store = VectorStore(db=db_session)

        results = vector_store.similarity_search(
            query="What is IRR?", k=3, filter_metadata={"fund_id": test_fund.id}
        )

        # Should return results only for the specific fund
        for result in results:
            assert result["fund_id"] == test_fund.id

    def test_05_similarity_search_with_threshold(self, db_session):
        """Test similarity search with threshold"""
        vector_store = VectorStore(db=db_session)

        results = vector_store.similarity_search(
            query="Explain paid-in capital", k=5, similarity_threshold=0.7
        )

        # All results should meet the threshold
        for result in results:
            assert result["score"] >= 0.7

    @pytest.mark.asyncio
    async def test_06_query_engine_definition_query(self, db_session, test_fund):
        """Test query engine with definition query"""
        query_engine = QueryEngine(db=db_session)

        response = await query_engine.process_query(
            query="What does DPI mean?", fund_id=test_fund.id
        )

        assert "answer" in response
        assert response["intent"] == "definition"
        assert len(response["sources"]) > 0
        assert response["processing_time"] > 0

        # Answer should mention distributions or paid-in
        answer_lower = response["answer"].lower()
        assert "distribution" in answer_lower or "paid" in answer_lower

    @pytest.mark.asyncio
    async def test_07_query_engine_with_metrics(self, db_session, test_fund):
        """Test query engine calculates metrics for calculation queries"""
        # First, add some sample transactions to the fund
        from datetime import datetime

        from app.models import CapitalCall, Distribution

        capital_call = CapitalCall(
            fund_id=test_fund.id,
            call_date=datetime(2024, 1, 15),
            call_type="Call 1",
            amount=10000000.00,
            description="Initial capital call",
        )
        distribution = Distribution(
            fund_id=test_fund.id,
            distribution_date=datetime(2024, 6, 15),
            distribution_type="Return of Capital",
            amount=4000000.00,
            is_recallable=False,
            description="First distribution",
        )

        db_session.add(capital_call)
        db_session.add(distribution)
        db_session.commit()

        # Now query for DPI
        query_engine = QueryEngine(db=db_session)

        response = await query_engine.process_query(
            query="What is the current DPI for this fund?", fund_id=test_fund.id
        )

        assert response["intent"] == "calculation"
        assert "metrics" in response
        assert response["metrics"] is not None

        # DPI should be approximately 0.4 (4M / 10M)
        if "dpi" in response["metrics"]:
            assert 0.35 <= response["metrics"]["dpi"] <= 0.45

    @pytest.mark.asyncio
    async def test_08_query_engine_conversation_history(self, db_session, test_fund):
        """Test query engine with conversation history"""
        query_engine = QueryEngine(db=db_session)

        # First query
        response1 = await query_engine.process_query(
            query="What is DPI?", fund_id=test_fund.id
        )

        # Follow-up query with history
        conversation_history = [
            {"role": "user", "content": "What is DPI?"},
            {"role": "assistant", "content": response1["answer"]},
        ]

        response2 = await query_engine.process_query(
            query="How is it calculated?",
            fund_id=test_fund.id,
            conversation_history=conversation_history,
        )

        assert "answer" in response2
        # Should understand "it" refers to DPI from context

    def test_09_vector_store_clear_by_fund(self, db_session, test_fund):
        """Test clearing vector store for specific fund"""
        vector_store = VectorStore(db=db_session)

        # Get initial count
        initial_count = vector_store.get_document_count(fund_id=test_fund.id)

        # Clear fund-specific data
        vector_store.clear(fund_id=test_fund.id)

        # Verify deletion
        final_count = vector_store.get_document_count(fund_id=test_fund.id)
        assert final_count < initial_count or final_count == 0

    @pytest.mark.asyncio
    async def test_10_end_to_end_rag_flow(
        self, db_session, test_fund, sample_document_chunks
    ):
        """Test complete end-to-end RAG flow"""
        # Step 1: Add documents to vector store
        vector_store = VectorStore(db=db_session)
        count = vector_store.add_documents_batch(sample_document_chunks)
        assert count > 0

        # Step 2: Verify semantic search works
        search_results = vector_store.similarity_search(
            query="What is DPI?", k=3, filter_metadata={"fund_id": test_fund.id}
        )
        assert len(search_results) > 0

        # Step 3: Process query through RAG engine
        query_engine = QueryEngine(db=db_session)
        response = await query_engine.process_query(
            query="Explain DPI and how it's calculated", fund_id=test_fund.id
        )

        # Step 4: Verify response quality
        assert "answer" in response
        assert response["intent"] in ["definition", "calculation"]
        assert len(response["sources"]) > 0
        assert response["sources"][0]["score"] > 0.5

        # Answer should be substantive
        assert len(response["answer"]) > 50


@pytest.mark.integration
class TestDocumentProcessorVectorIntegration:
    """Test document processor integration with vector store"""

    @pytest.fixture
    def db_session(self):
        """Get database session"""
        db = SessionLocal()
        yield db
        db.close()

    @pytest.fixture
    def test_fund(self, db_session):
        """Create test fund"""
        from app.models import Fund

        fund = Fund(
            name="Doc Processor Test Fund",
            gp_name="Test GP",
            fund_type="VC",
            vintage_year=2024,
        )
        db_session.add(fund)
        db_session.commit()
        db_session.refresh(fund)

        yield fund

        db_session.query(Fund).filter(Fund.id == fund.id).delete()
        db_session.commit()

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="Requires API key for embeddings (GOOGLE_API_KEY or OPENAI_API_KEY)",
    )
    def test_text_chunks_stored_to_vector_db(self, db_session, test_fund):
        """Test that document processor stores text chunks to vector DB"""

        # Create sample text chunks
        text_chunks = [
            {
                "text": "Sample financial document content about DPI metrics",
                "metadata": {
                    "document_id": 999,
                    "fund_id": test_fund.id,
                    "chunk_type": "general",
                },
            }
        ]

        try:
            # Initialize document processor with db session
            doc_processor = DocumentProcessor(db=db_session)

            # Store chunks
            stored_count = doc_processor._store_chunks_to_vector_db(text_chunks)

            assert stored_count == 1

            # Verify in vector store
            vector_store = VectorStore(db=db_session)
            results = vector_store.similarity_search(
                query="DPI metrics", k=5, filter_metadata={"fund_id": test_fund.id}
            )

            assert len(results) > 0
            assert any("DPI" in r["content"] for r in results)
        except Exception as e:
            # If rate limited or API error, skip
            error_msg = str(e).lower()
            if "429" in str(e) or "rate" in error_msg or "quota" in error_msg:
                pytest.skip(f"API rate limited or quota exceeded: {e}")
            raise


@pytest.mark.integration
class TestRAGPerformance:
    """Test RAG system performance characteristics"""

    @pytest.fixture
    def db_session(self):
        """Get database session"""
        db = SessionLocal()
        yield db
        db.close()

    @pytest.mark.asyncio
    async def test_query_response_time(self, db_session):
        """Test that queries complete in reasonable time"""
        query_engine = QueryEngine(db=db_session)

        response = await query_engine.process_query(query="What is DPI?")

        # Should complete in under 10 seconds
        assert response["processing_time"] < 10.0

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="Requires API key for embeddings (GOOGLE_API_KEY or OPENAI_API_KEY)",
    )
    def test_batch_embedding_performance(self, db_session):
        """Test batch embedding performance"""
        import time

        try:
            vector_store = VectorStore(db=db_session)

            # Create 100 test documents
            documents = [
                {
                    "text": f"Test document number {i} about fund performance metrics",
                    "metadata": {"document_id": 1000 + i, "fund_id": 1},
                }
                for i in range(100)
            ]

            start_time = time.time()
            count = vector_store.add_documents_batch(documents, batch_size=50)
            elapsed = time.time() - start_time

            assert count == 100
            # Should process 100 docs in under 60 seconds
            assert elapsed < 60.0

            # Cleanup
            vector_store.clear(fund_id=1)
        except Exception as e:
            # If rate limited or API error, skip
            error_msg = str(e).lower()
            if "429" in str(e) or "rate" in error_msg or "quota" in error_msg:
                pytest.skip(f"API rate limited or quota exceeded: {e}")
            raise
