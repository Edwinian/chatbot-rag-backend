import os
from typing import Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from sympy import re

from chroma_service import ChromaService
from db_service import DBService
from pydantic_models import ModelName, QueryInput

load_dotenv()  # Loads the .env file


class LangChainService:
    def __init__(
        self,
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
        self.chroma_service = ChromaService()
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

        self.llm = self._initialize_llm()

    def _initialize_llm(self):
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

    def get_model_name(self):
        """Return the model name."""
        return self.model_name

    def get_huggingface_llm(self):
        """Return the preloaded ChatHuggingFace LLM."""
        return self.llm

    def get_rag_chain(self, collection_name: str = None):
        """
        Create and return a RAG chain for the specified collection.

        Returns:
            RAG chain for processing queries
        """
        retriever = self.chroma_service.get_retriever(
            collection_name=collection_name, search_kwargs={"k": 2}
        )
        # A history-aware retriever that rephrases the question if it depends on past messages (e.g., if the user says “Tell me more,” it figures out what “more” means by looking at the history)
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )
        # A question-answer chain that combines the retrieved documents, chat history, and question to produce a clear answer
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        return rag_chain

    def get_model_answer(
        self,
        query_input: QueryInput,
        collection_name: str = None,
        session_id: str = None,
    ):
        chat_history = (
            self.db_service.get_chat_history(session_id) if session_id else []
        )
        rag_chain = self.get_rag_chain(collection_name=collection_name)
        answer = rag_chain.invoke(
            {"input": query_input.question, "chat_history": chat_history}
        )["answer"]
        return answer
