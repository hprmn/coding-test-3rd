"""
Query Engine Service for RAG-based Question Answering

This module implements the RAG (Retrieval Augmented Generation) pipeline for
answering questions about fund performance. It combines semantic search from
the vector store with structured data from SQL to provide accurate answers.

Key Features:
- Intent classification (calculation, definition, retrieval, general)
- Context retrieval from vector store
- Integration with metrics calculator
- Advanced prompt engineering for accurate responses
- Source citation and transparency
"""
from typing import Dict, Any, List, Optional
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
from app.services.vector_store import VectorStore
from app.services.metrics_calculator import MetricsCalculator
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    RAG-based query engine for fund analysis

    This class orchestrates the entire question-answering pipeline:
    1. Classify user intent
    2. Retrieve relevant context from vector store
    3. Calculate metrics if needed
    4. Generate accurate response using LLM

    Example:
        >>> query_engine = QueryEngine(db_session)
        >>> response = await query_engine.process_query(
        ...     query="What is the current DPI?",
        ...     fund_id=1
        ... )
    """

    def __init__(self, db: Session):
        """
        Initialize query engine

        Args:
            db: Database session for metrics calculation
        """
        self.db = db
        self.vector_store = VectorStore(db)
        self.metrics_calculator = MetricsCalculator(db)
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """
        Initialize LLM based on configuration

        Priority:
        1. OpenAI GPT (if API key configured)
        2. Google Gemini (if API key configured)
        3. Local Ollama (fallback)

        Returns:
            LLM instance
        """
        try:
            if settings.OPENAI_API_KEY:
                logger.info("Initializing OpenAI LLM")
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0,  # Deterministic for financial data
                    openai_api_key=settings.OPENAI_API_KEY
                )
            elif settings.GOOGLE_API_KEY:
                logger.info("Initializing Google Gemini LLM")
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    temperature=0,
                    google_api_key=settings.GOOGLE_API_KEY
                )
            else:
                logger.info("Initializing local Ollama LLM")
                return Ollama(
                    model="llama3.2",
                    temperature=0
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    async def process_query(
        self,
        query: str,
        fund_id: Optional[int] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process user query using RAG pipeline

        This is the main entry point for question answering.

        Args:
            query: User's question
            fund_id: Optional fund ID for context filtering
            conversation_history: Previous conversation messages

        Returns:
            Dictionary containing:
            - answer: Generated response
            - sources: Retrieved documents with scores
            - metrics: Calculated metrics (if applicable)
            - intent: Classified intent
            - processing_time: Time taken in seconds

        Example:
            >>> response = await process_query("What is DPI?", fund_id=1)
            >>> print(response['answer'])
        """
        start_time = time.time()

        try:
            # Step 1: Classify query intent
            intent = self._classify_intent(query)
            logger.info(f"Query intent classified as: {intent}")

            # Step 2: Retrieve relevant context from vector store
            filter_metadata = {"fund_id": fund_id} if fund_id else None
            relevant_docs = self.vector_store.similarity_search(
                query=query,
                k=settings.TOP_K_RESULTS,
                filter_metadata=filter_metadata,
                similarity_threshold=settings.SIMILARITY_THRESHOLD
            )

            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

            # Step 3: Calculate metrics if needed
            metrics = None
            metrics_breakdown = None

            if intent == "calculation" and fund_id:
                try:
                    metrics = self.metrics_calculator.calculate_all_metrics(fund_id)
                    # Get detailed breakdown for transparency
                    metrics_breakdown = self.metrics_calculator.get_calculation_breakdown(fund_id)
                    logger.info(f"Calculated metrics for fund {fund_id}")
                except Exception as e:
                    logger.warning(f"Failed to calculate metrics: {e}")

            # Step 4: Generate response using LLM with context
            answer = await self._generate_response(
                query=query,
                intent=intent,
                context=relevant_docs,
                metrics=metrics,
                metrics_breakdown=metrics_breakdown,
                conversation_history=conversation_history or []
            )

            processing_time = time.time() - start_time

            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "score": round(doc.get("score", 0), 3),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in relevant_docs[:3]  # Return top 3 sources
                ],
                "metrics": metrics,
                "intent": intent,
                "processing_time": round(processing_time, 2)
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": [],
                "metrics": None,
                "intent": "error",
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e)
            }

    def _classify_intent(self, query: str) -> str:
        """
        Classify user query intent using keyword matching

        Intents:
        - calculation: Calculate metrics (DPI, IRR, etc.)
        - definition: Explain financial terms
        - retrieval: Find specific data (capital calls, distributions)
        - comparison: Compare funds or metrics
        - general: General questions

        Args:
            query: User's question

        Returns:
            Intent category string
        """
        query_lower = query.lower()

        # Comparison keywords (check first)
        comp_keywords = [
            "compare", "versus", "vs", "difference between",
            "better than", "worse than", "higher than", "lower than",
            "which fund"
        ]
        if any(keyword in query_lower for keyword in comp_keywords):
            return "comparison"

        # Calculation keywords - check for specific calculation requests
        # Look for patterns that request computed values
        calc_indicators = [
            "calculate", "current dpi", "current irr", "total pic",
            "what is the current", "what is the total", "what's the current",
            "what's the total", "how much is", "how much has",
            "show me the dpi", "show me the irr", "compute"
        ]

        # Check for calculation with fund reference
        has_fund_ref = any(f in query_lower for f in ["fund", "for fund"])
        has_metric = any(m in query_lower for m in ["dpi", "irr", "pic", "tvpi", "multiple"])

        # If asking "what is [metric] for [fund]" it's a calculation
        if has_fund_ref and has_metric:
            if any(calc in query_lower for calc in ["what is the", "what's the"]):
                return "calculation"

        # Check explicit calculation patterns
        if any(pattern in query_lower for pattern in calc_indicators):
            return "calculation"

        # Definition keywords - asking about what a term means
        def_patterns = [
            "what does", "what is a", "what are", "what is paid",
            "what is dpi", "what is irr", "what is tvpi",
            "mean", "define", "definition", "explain", "tell me about"
        ]

        # Financial terms that definitions usually ask about
        financial_terms = [
            "dpi", "irr", "tvpi", "pic", "paid-in capital", "paid in capital",
            "capital call", "distribution", "recallable", "gp", "lp",
            "carried interest", "hurdle rate", "waterfall", "clawback",
            "commitment", "drawdown", "multiple"
        ]

        # For "what is X" queries, check if it's asking for a definition or calculation
        if "what is" in query_lower or "what's" in query_lower:
            # If it has fund reference and metric, it's likely calculation
            if has_fund_ref and has_metric:
                return "calculation"
            # Otherwise check if it's asking about a term/concept
            elif has_metric or any(term in query_lower for term in financial_terms):
                return "definition"

        # Check other definition patterns - but only for financial terms
        for keyword in def_patterns:
            if keyword in query_lower:
                # Check if it's about a financial term
                has_financial_term = any(term in query_lower for term in financial_terms)
                # Exclude if it's clearly asking for calculation
                is_calculation = any(calc in query_lower for calc in ["current", "total", "for fund", "calculate"])

                # Special handling for "tell me about"
                if keyword == "tell me about":
                    # Only classify as definition if it mentions a financial term
                    if has_financial_term:
                        return "definition"
                    # Otherwise it's general (e.g., "Tell me about the fund strategy")
                    continue

                # For other keywords, check financial term
                if has_financial_term and not is_calculation:
                    return "definition"

        # Retrieval keywords - looking for data/records
        ret_keywords = [
            "show me", "list", "find", "search",
            "when was", "how many", "give me"
        ]
        # Check for retrieval patterns
        for keyword in ret_keywords:
            if keyword in query_lower:
                # Make sure it's not a definition or general question
                if keyword in ["show me", "give me"] and "all" not in query_lower:
                    # "show me the dpi" vs "show me all capital calls"
                    # If it has calculation indicators, skip
                    if any(m in query_lower for m in ["dpi", "irr", "current", "total"]):
                        continue
                return "retrieval"

        return "general"

    async def _generate_response(
        self,
        query: str,
        intent: str,
        context: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]],
        metrics_breakdown: Optional[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate response using LLM with retrieved context

        This method implements advanced prompt engineering for high-quality responses.

        Args:
            query: User's question
            intent: Classified intent
            context: Retrieved documents from vector store
            metrics: Calculated metrics (if applicable)
            metrics_breakdown: Detailed calculation breakdown
            conversation_history: Previous messages

        Returns:
            Generated answer string
        """
        # Build context string from top documents
        context_str = ""
        if context:
            context_str = "\n\n".join([
                f"[Document {i+1}] (relevance: {doc.get('score', 0):.2f})\n{doc['content']}"
                for i, doc in enumerate(context[:5])  # Use top 5 documents
            ])
        else:
            context_str = "No relevant documents found in the knowledge base."

        # Build metrics string with detailed breakdown
        metrics_str = ""
        if metrics:
            metrics_str = "\n\n### Available Metrics:\n"
            for key, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        metrics_str += f"- **{key.upper()}**: {value:.4f}\n"
                    else:
                        metrics_str += f"- **{key.upper()}**: {value}\n"

            # Add calculation breakdown for transparency
            if metrics_breakdown and intent == "calculation":
                metrics_str += "\n### Calculation Details:\n"
                if "dpi_calculation" in metrics_breakdown:
                    dpi_calc = metrics_breakdown["dpi_calculation"]
                    metrics_str += f"- Total Distributions: ${dpi_calc.get('total_distributions', 0):,.2f}\n"
                    metrics_str += f"- Paid-In Capital: ${dpi_calc.get('paid_in_capital', 0):,.2f}\n"
                    metrics_str += f"- DPI Formula: {dpi_calc.get('total_distributions', 0):,.2f} / {dpi_calc.get('paid_in_capital', 1):,.2f}\n"

        # Build conversation history string
        history_str = ""
        if conversation_history:
            history_str = "\n\n### Previous Conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_str += f"{role.capitalize()}: {content}\n"

        # Create prompt based on intent
        if intent == "calculation":
            system_prompt = """You are a financial analyst assistant specializing in private equity fund performance metrics.

Your role:
- Calculate and explain fund performance metrics (DPI, IRR, TVPI, RVPI, PIC)
- Use the provided metrics data and calculation details
- Show your work step-by-step
- Explain what the numbers mean in practical terms
- Always cite the data sources you use

Guidelines for calculations:
- Use ONLY the metrics provided in the context
- Show the formula used
- Explain what the result means for LPs (Limited Partners)
- If data is missing, clearly state what's needed
- Round numbers appropriately (2-4 decimal places)

Format:
1. Direct answer with the calculated value
2. Formula and calculation steps
3. Interpretation of what this means
4. Any caveats or assumptions"""

        elif intent == "definition":
            system_prompt = """You are a financial education assistant specializing in private equity terminology.

Your role:
- Explain financial terms in clear, simple language
- Use examples to illustrate concepts
- Reference the provided documents for accurate definitions
- Relate concepts to real-world fund operations

Guidelines:
- Start with a simple definition
- Provide a more detailed explanation
- Give a practical example
- Mention why it matters to LPs
- Always cite your sources from the documents"""

        elif intent == "retrieval":
            system_prompt = """You are a data analyst assistant helping users find specific information about fund transactions.

Your role:
- Find and present specific data from documents
- Organize information clearly (use tables or lists)
- Highlight key figures
- Provide context for the numbers

Guidelines:
- Extract exact data from the context
- Present in an organized format
- Include dates, amounts, and descriptions
- Summarize key findings
- Cite source documents"""

        else:
            system_prompt = """You are a knowledgeable financial analyst assistant for private equity fund analysis.

Your role:
- Answer questions about fund performance using provided context
- Combine information from multiple sources when needed
- Explain complex concepts clearly
- Always ground your answers in the provided data

Guidelines:
- Be accurate and precise
- Use data from the context
- Explain your reasoning
- Admit when you don't have enough information
- Suggest what additional data would be helpful"""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", """### User Question:
{query}

### Intent: {intent}

### Retrieved Context:
{context}
{metrics}
{history}

### Instructions:
Please provide a comprehensive answer to the user's question based on the context and metrics provided above.
- Be accurate and cite your sources
- Use clear, professional language
- Include specific numbers and calculations where applicable
- If you're unsure or lack information, say so

Answer:""")
        ])

        # Generate response
        try:
            messages = prompt.format_messages(
                query=query,
                intent=intent,
                context=context_str,
                metrics=metrics_str,
                history=history_str
            )

            response = self.llm.invoke(messages)

            if hasattr(response, 'content'):
                return response.content
            return str(response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating a response. Please try rephrasing your question. Error: {str(e)}"

    def get_suggested_questions(self, fund_id: Optional[int] = None) -> List[str]:
        """
        Get suggested questions based on available data

        Args:
            fund_id: Optional fund ID to customize suggestions

        Returns:
            List of suggested question strings
        """
        suggestions = [
            "What is DPI and how is it calculated?",
            "What does Paid-In Capital mean?",
            "Explain the difference between DPI and IRR",
            "What is a recallable distribution?",
        ]

        if fund_id:
            suggestions.extend([
                f"What is the current DPI for fund {fund_id}?",
                f"Calculate the IRR for fund {fund_id}",
                f"Show me all capital calls for fund {fund_id}",
                f"How is fund {fund_id} performing?"
            ])

        return suggestions
