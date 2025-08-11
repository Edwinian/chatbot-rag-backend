import os
from typing import Any, Dict, List, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import requests

from chroma_service import ChromaService
from db_service import DBService
from pydantic_models import ModelName

load_dotenv()  # Loads the .env file


class LangChainService:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = ModelName.Mixtral_v0_1.value,
        max_length: int = 512,
    ):
        """
        Initialize LangChainService with a ChromaService instance and LLM configuration.

        Args:
            chroma_service: ChromaService instance for retrieval
            model_name: Hugging Face model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)
            max_length: Maximum length for generated text
        """
        self.chroma_service = ChromaService(collection_name=collection_name)
        self.db_service = DBService()
        self.output_parser = StrOutputParser()
        self.model_name = model_name
        self.max_length = max_length

        # Initialize prompt templates
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Use the following context to answer the user's question.\n\nContext: {context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        self.chat_llm = self._initialize_chat_llm()

    def _initialize_chat_llm(self):
        """Initialize and return the ChatHuggingFace LLM."""
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name,
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
                max_new_tokens=self.max_length,
                temperature=0.7,  # Controls response randomness
                top_p=0.9,  # Controls response diversity
                return_full_text=False,
            )
            # Wrap with ChatHuggingFace since HuggingFaceEndpoint always defaults to text-generation task, which is not supported by all models
            return ChatHuggingFace(llm=endpoint)
        except Exception as e:
            raise Exception(f"Failed to initialize LLM {self.model_name}: {str(e)}")

    def get_rag_chain(self):
        """
        Create and return a RAG chain for the specified collection.

        Returns:
            RAG chain for processing queries
        """
        retriever = self.chroma_service.get_retriever(search_kwargs={"k": 2})
        # A history-aware retriever that rephrases the question if it depends on past messages (e.g., if the user says “Tell me more,” it figures out what “more” means by looking at the history)
        history_aware_retriever = create_history_aware_retriever(
            self.chat_llm, retriever, self.contextualize_q_prompt
        )
        # A question-answer chain that combines the retrieved documents, chat history, and question to produce a clear answer
        question_answer_chain = create_stuff_documents_chain(
            self.chat_llm, self.qa_prompt
        )
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        return rag_chain

    def get_model_answer(
        self,
        query: str,
        session_id: str = None,
        skip_hybrid: bool = False,
    ):
        chat_history = (
            self.db_service.get_chat_history(session_id) if session_id else []
        )
        rag_chain = self.get_rag_chain()
        rag_results = rag_chain.invoke({"input": query, "chat_history": chat_history})
        answer = rag_results["answer"]

        if not skip_hybrid:
            need_hybrid = self.need_hybrid_answer(query=query, answer=answer)
            print("need_hybrid", need_hybrid)

            if need_hybrid:
                answer = self.get_hybrid_answer(
                    query=query, chat_history=chat_history, rag_results=rag_results
                )

        return answer

    def retrieve_content(self, query: str) -> str:
        try:
            api_url = "https://serpapi.com/search"
            headers = {"Content-Type": "application/json"}
            params = {
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "engine": "google",
                "num": 10,
            }
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            result = ""
            if "answer_box" in data and "answer" in data["answer_box"]:
                result = data["answer_box"]["answer"]

            if "organic_results" in data and len(data["organic_results"]) > 0:
                snippets = [
                    result.get("snippet", None)
                    for result in data["organic_results"]
                    if result.get("snippet")
                ]
                valid_snippets = [s for s in snippets if s]
                result = ". ".join(valid_snippets) + (". " if valid_snippets else "")

            return result
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve content from web: {str(e)}")
            return ""

    def get_hybrid_answer(self, query: str, chat_history: List[Dict], rag_results: Any):
        search_results = self.retrieve_content(query)

        if not search_results:
            return rag_results["answer"]

        # Combine both sources of information
        combined_context = f"""
        Vector Database Results:
        {rag_results['context']}
        
        Web Search Results:
        {search_results}
        """

        # Create a prompt for the combined context
        hybrid_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Use the following information to answer the user's question."
                    "\n\n{context}\n\n"
                    "If the information is conflicting, prioritize the most recent or authoritative source.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        hybrid_chain = hybrid_prompt | self.chat_llm | self.output_parser
        answer = hybrid_chain.invoke(
            {"input": query, "chat_history": chat_history, "context": combined_context}
        )

        return answer

    def need_hybrid_answer(self, query: str, answer: str) -> bool:
        """
        Determine if a hybrid answer is needed based on the query.

        """

        def _is_low_confident(answer: str) -> bool:
            """Heuristic to detect when external search might be needed"""
            low_confidence_phrases = [
                "I don't know",
                "I'm not sure",
                "not in my knowledge",
                "no information",
                "can't find",
                "unable to",
                "you may want to check",
                "verify this information",
                "as of my last update",
                "latest",
                "news",
                "update",
                "current",
            ]
            return any(phrase in answer.lower() for phrase in low_confidence_phrases)

        def _llm_decides(query: str, answer: str) -> bool:
            """Use the LLM to evaluate if external search is needed"""
            decision_prompt = f"""
            Based on the following interaction, should we search external sources for more information?
            
            Query: {query}
            Current Answer: {answer}
            
            Respond ONLY with 'YES' or 'NO'. Do not provide any explanation.
            """

            response = self.get_model_answer(
                query=decision_prompt,
                skip_hybrid=True,
            )

            return response.upper() == "YES"

        return any([_is_low_confident(answer), _llm_decides(query, answer)])
