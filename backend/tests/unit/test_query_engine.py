"""
Unit tests for Query Engine (RAG System)

Tests cover:
- Intent classification
- Context retrieval
- Prompt engineering
- LLM integration
- End-to-end query processing
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.query_engine import QueryEngine


class TestIntentClassification:
    """Test query intent classification"""

    @pytest.fixture
    def query_engine(self):
        """Create query engine with mocked dependencies"""
        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore'), \
             patch('app.services.query_engine.MetricsCalculator'), \
             patch('app.services.query_engine.settings') as mock_settings:

            mock_settings.OPENAI_API_KEY = "sk-test"
            mock_settings.OPENAI_MODEL = "gpt-4"

            with patch('app.services.query_engine.ChatOpenAI'):
                return QueryEngine(db=mock_db)

    def test_classify_calculation_intent(self, query_engine):
        """Test classification of calculation queries"""
        calculation_queries = [
            "What is the current DPI?",
            "Calculate the IRR for this fund",
            "What is the DPI for fund 1?",
            "Show me the current IRR",
            "How much is the total PIC?",
        ]

        for query in calculation_queries:
            intent = query_engine._classify_intent(query)
            assert intent == "calculation", f"Failed for query: {query}"

    def test_classify_definition_intent(self, query_engine):
        """Test classification of definition queries"""
        definition_queries = [
            "What does DPI mean?",
            "What is Paid-In Capital?",
            "Explain IRR to me",
            "Define recallable distribution",
            "What is a capital call?",
            "Tell me about TVPI",
        ]

        for query in definition_queries:
            intent = query_engine._classify_intent(query)
            assert intent == "definition", f"Failed for query: {query}"

    def test_classify_retrieval_intent(self, query_engine):
        """Test classification of retrieval queries"""
        retrieval_queries = [
            "Show me all capital calls in 2024",
            "List all distributions",
            "Find capital calls for this fund",
            "When was the last distribution?",
            "How many capital calls were made?",
            "Give me all adjustments",
        ]

        for query in retrieval_queries:
            intent = query_engine._classify_intent(query)
            assert intent == "retrieval", f"Failed for query: {query}"

    def test_classify_comparison_intent(self, query_engine):
        """Test classification of comparison queries"""
        comparison_queries = [
            "Compare fund A versus fund B",
            "Which fund has better DPI?",
            "Is this fund performing better than last year?",
            "What's the difference between DPI and IRR?",
        ]

        for query in comparison_queries:
            intent = query_engine._classify_intent(query)
            assert intent == "comparison", f"Failed for query: {query}"

    def test_classify_general_intent(self, query_engine):
        """Test classification of general queries"""
        general_queries = [
            "How are you?",
            "Tell me about the fund strategy",
            "What investments does this fund make?",
        ]

        for query in general_queries:
            intent = query_engine._classify_intent(query)
            assert intent == "general", f"Failed for query: {query}"


class TestQueryProcessing:
    """Test end-to-end query processing"""

    @pytest.fixture
    def mock_query_engine(self):
        """Create fully mocked query engine"""
        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore') as mock_vs_class, \
             patch('app.services.query_engine.MetricsCalculator') as mock_mc_class, \
             patch('app.services.query_engine.settings') as mock_settings:

            mock_settings.OPENAI_API_KEY = "sk-test"
            mock_settings.OPENAI_MODEL = "gpt-4"
            mock_settings.TOP_K_RESULTS = 5
            mock_settings.SIMILARITY_THRESHOLD = 0.7

            # Mock vector store
            mock_vs = Mock()
            mock_vs.similarity_search = Mock(return_value=[
                {
                    "id": 1,
                    "content": "DPI (Distributions to Paid-In Capital) is a key metric...",
                    "score": 0.95,
                    "metadata": {"section": "definitions"}
                },
                {
                    "id": 2,
                    "content": "The formula for DPI is: Total Distributions / Paid-In Capital",
                    "score": 0.87,
                    "metadata": {"section": "formulas"}
                }
            ])
            mock_vs_class.return_value = mock_vs

            # Mock metrics calculator
            mock_mc = Mock()
            mock_mc.calculate_all_metrics = Mock(return_value={
                "dpi": 0.4000,
                "irr": 0.1250,
                "pic": 10000000.00
            })
            mock_mc.get_calculation_breakdown = Mock(return_value={
                "dpi_calculation": {
                    "total_distributions": 4000000.00,
                    "paid_in_capital": 10000000.00
                }
            })
            mock_mc_class.return_value = mock_mc

            # Mock LLM
            with patch('app.services.query_engine.ChatOpenAI') as mock_llm_class:
                mock_llm = Mock()
                mock_response = Mock()
                mock_response.content = "The current DPI is 0.4000, which means..."
                mock_llm.invoke = Mock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                qe = QueryEngine(db=mock_db)
                qe.llm = mock_llm
                qe.vector_store = mock_vs
                qe.metrics_calculator = mock_mc

                return qe

    @pytest.mark.asyncio
    async def test_process_definition_query(self, mock_query_engine):
        """Test processing a definition query"""
        response = await mock_query_engine.process_query(
            query="What is DPI?",
            fund_id=1
        )

        assert "answer" in response
        assert "sources" in response
        assert "intent" in response
        assert response["intent"] == "definition"
        assert len(response["sources"]) > 0
        assert response["sources"][0]["score"] > 0.8

    @pytest.mark.asyncio
    async def test_process_calculation_query(self, mock_query_engine):
        """Test processing a calculation query"""
        response = await mock_query_engine.process_query(
            query="What is the current DPI?",
            fund_id=1
        )

        assert "answer" in response
        assert "sources" in response
        assert "metrics" in response
        assert "intent" in response
        assert response["intent"] == "calculation"
        assert response["metrics"] is not None
        assert "dpi" in response["metrics"]

    @pytest.mark.asyncio
    async def test_process_query_without_fund_id(self, mock_query_engine):
        """Test processing query without fund_id"""
        response = await mock_query_engine.process_query(
            query="What does IRR mean?"
        )

        assert "answer" in response
        assert "sources" in response
        # Should still work, but metrics might be None
        assert response["intent"] == "definition"

    @pytest.mark.asyncio
    async def test_process_query_with_conversation_history(self, mock_query_engine):
        """Test processing query with conversation history"""
        conversation_history = [
            {"role": "user", "content": "What is DPI?"},
            {"role": "assistant", "content": "DPI is Distributions to Paid-In Capital..."}
        ]

        response = await mock_query_engine.process_query(
            query="How is it calculated?",
            fund_id=1,
            conversation_history=conversation_history
        )

        assert "answer" in response
        assert "processing_time" in response
        assert response["processing_time"] >= 0

    @pytest.mark.asyncio
    async def test_process_query_error_handling(self):
        """Test error handling in query processing"""
        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore') as mock_vs_class, \
             patch('app.services.query_engine.settings') as mock_settings:

            mock_settings.OPENAI_API_KEY = "sk-test"

            # Mock vector store to raise error
            mock_vs = Mock()
            mock_vs.similarity_search = Mock(side_effect=Exception("Search failed"))
            mock_vs_class.return_value = mock_vs

            with patch('app.services.query_engine.ChatOpenAI'), \
                 patch('app.services.query_engine.MetricsCalculator'):

                qe = QueryEngine(db=mock_db)
                response = await qe.process_query("test query")

                assert "error" in response
                assert response["intent"] == "error"


class TestPromptEngineering:
    """Test prompt engineering for different intents"""

    @pytest.fixture
    def mock_query_engine(self):
        """Create query engine with mocked LLM"""
        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore'), \
             patch('app.services.query_engine.MetricsCalculator'), \
             patch('app.services.query_engine.settings') as mock_settings:

            mock_settings.OPENAI_API_KEY = "sk-test"

            with patch('app.services.query_engine.ChatOpenAI') as mock_llm_class:
                mock_llm = Mock()
                mock_response = Mock()
                mock_response.content = "Test response"
                mock_llm.invoke = Mock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                qe = QueryEngine(db=mock_db)
                qe.llm = mock_llm
                return qe

    @pytest.mark.asyncio
    async def test_calculation_prompt_includes_metrics(self, mock_query_engine):
        """Test that calculation prompts include metrics data"""
        context = [{"content": "test", "score": 0.9}]
        metrics = {"dpi": 0.4, "irr": 0.12}
        metrics_breakdown = {
            "dpi_calculation": {
                "total_distributions": 4000000,
                "paid_in_capital": 10000000
            }
        }

        response = await mock_query_engine._generate_response(
            query="What is DPI?",
            intent="calculation",
            context=context,
            metrics=metrics,
            metrics_breakdown=metrics_breakdown,
            conversation_history=[]
        )

        # Verify LLM was called
        assert mock_query_engine.llm.invoke.called

        # Check that prompt includes metrics
        call_args = mock_query_engine.llm.invoke.call_args[0][0]
        prompt_text = str(call_args)
        assert "0.4" in prompt_text or "DPI" in prompt_text.upper()

    @pytest.mark.asyncio
    async def test_definition_prompt_includes_context(self, mock_query_engine):
        """Test that definition prompts include retrieved context"""
        context = [
            {"content": "DPI is a key metric for fund performance", "score": 0.95}
        ]

        response = await mock_query_engine._generate_response(
            query="What is DPI?",
            intent="definition",
            context=context,
            metrics=None,
            metrics_breakdown=None,
            conversation_history=[]
        )

        assert mock_query_engine.llm.invoke.called

        call_args = mock_query_engine.llm.invoke.call_args[0][0]
        prompt_text = str(call_args)
        assert "DPI" in prompt_text

    @pytest.mark.asyncio
    async def test_retrieval_prompt_structure(self, mock_query_engine):
        """Test retrieval intent prompt structure"""
        context = [
            {"content": "Capital call on 2024-01-15: $5M", "score": 0.9}
        ]

        response = await mock_query_engine._generate_response(
            query="Show me capital calls",
            intent="retrieval",
            context=context,
            metrics=None,
            metrics_breakdown=None,
            conversation_history=[]
        )

        assert response == "Test response"
        assert mock_query_engine.llm.invoke.called


class TestLLMIntegration:
    """Test LLM initialization and integration"""

    @patch('app.services.query_engine.settings')
    @patch('app.services.query_engine.ChatOpenAI')
    def test_initialize_with_openai(self, mock_openai, mock_settings):
        """Test LLM initialization with OpenAI"""
        mock_settings.OPENAI_API_KEY = "sk-test"
        mock_settings.OPENAI_MODEL = "gpt-4-turbo-preview"

        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore'), \
             patch('app.services.query_engine.MetricsCalculator'):

            qe = QueryEngine(db=mock_db)

            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs['temperature'] == 0  # Deterministic for financial data

    @patch('app.services.query_engine.settings')
    @patch('app.services.query_engine.Ollama')
    def test_initialize_with_ollama(self, mock_ollama, mock_settings):
        """Test LLM initialization with Ollama (fallback)"""
        mock_settings.OPENAI_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.GEMINI_MODEL = "gemini-pro"

        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore'), \
             patch('app.services.query_engine.MetricsCalculator'):

            qe = QueryEngine(db=mock_db)

            mock_ollama.assert_called_once()


class TestSuggestedQuestions:
    """Test suggested questions feature"""

    @pytest.fixture
    def query_engine(self):
        """Create query engine"""
        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore'), \
             patch('app.services.query_engine.MetricsCalculator'), \
             patch('app.services.query_engine.settings') as mock_settings:

            mock_settings.OPENAI_API_KEY = "sk-test"

            with patch('app.services.query_engine.ChatOpenAI'):
                return QueryEngine(db=mock_db)

    def test_suggested_questions_without_fund(self, query_engine):
        """Test getting suggested questions without fund_id"""
        suggestions = query_engine.get_suggested_questions()

        assert len(suggestions) > 0
        assert any("DPI" in s for s in suggestions)
        assert any("Paid-In Capital" in s for s in suggestions)

    def test_suggested_questions_with_fund(self, query_engine):
        """Test getting suggested questions with fund_id"""
        suggestions = query_engine.get_suggested_questions(fund_id=1)

        assert len(suggestions) > 0
        # Should include fund-specific questions
        assert any("fund 1" in s for s in suggestions)
        assert any("DPI" in s or "IRR" in s for s in suggestions)


class TestContextRetrieval:
    """Test context retrieval from vector store"""

    @pytest.mark.asyncio
    async def test_context_retrieval_with_fund_filter(self):
        """Test that fund_id filter is applied to vector search"""
        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore') as mock_vs_class, \
             patch('app.services.query_engine.MetricsCalculator'), \
             patch('app.services.query_engine.settings') as mock_settings:

            mock_settings.OPENAI_API_KEY = "sk-test"
            mock_settings.TOP_K_RESULTS = 5
            mock_settings.SIMILARITY_THRESHOLD = 0.7

            mock_vs = Mock()
            mock_vs.similarity_search = Mock(return_value=[])
            mock_vs_class.return_value = mock_vs

            with patch('app.services.query_engine.ChatOpenAI') as mock_llm_class:
                mock_llm = Mock()
                mock_llm.invoke = Mock(return_value=Mock(content="response"))
                mock_llm_class.return_value = mock_llm

                qe = QueryEngine(db=mock_db)
                qe.llm = mock_llm
                qe.vector_store = mock_vs

                await qe.process_query(query="test", fund_id=123)

                # Verify similarity_search was called with fund filter
                mock_vs.similarity_search.assert_called_once()
                call_kwargs = mock_vs.similarity_search.call_args[1]
                assert call_kwargs['filter_metadata'] == {"fund_id": 123}

    @pytest.mark.asyncio
    async def test_context_retrieval_respects_top_k(self):
        """Test that TOP_K_RESULTS setting is respected"""
        mock_db = Mock()

        with patch('app.services.query_engine.VectorStore') as mock_vs_class, \
             patch('app.services.query_engine.MetricsCalculator'), \
             patch('app.services.query_engine.settings') as mock_settings:

            mock_settings.OPENAI_API_KEY = "sk-test"
            mock_settings.TOP_K_RESULTS = 3
            mock_settings.SIMILARITY_THRESHOLD = 0.7

            mock_vs = Mock()
            mock_vs.similarity_search = Mock(return_value=[])
            mock_vs_class.return_value = mock_vs

            with patch('app.services.query_engine.ChatOpenAI') as mock_llm_class:
                mock_llm = Mock()
                mock_llm.invoke = Mock(return_value=Mock(content="response"))
                mock_llm_class.return_value = mock_llm

                qe = QueryEngine(db=mock_db)
                qe.llm = mock_llm
                qe.vector_store = mock_vs

                await qe.process_query(query="test")

                call_kwargs = mock_vs.similarity_search.call_args[1]
                assert call_kwargs['k'] == 3
