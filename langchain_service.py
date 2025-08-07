from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch


class LangChainService:
    def __init__(
        self,
        chroma_service,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        max_length: int = 512,
    ):
        """
        Initialize LangChainService with a ChromaService instance and LLM configuration.

        Args:
            chroma_service: ChromaService instance for retrieval
            model_name: Hugging Face model name for text generation
            max_length: Maximum length for generated text
        """
        self.chroma_service = chroma_service
        self.model_name = model_name
        self.max_length = max_length
        self.output_parser = StrOutputParser()

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
                ("system", f"[INST] {self.contextualize_q_system_prompt} [/INST]"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "[INST] You are a helpful AI assistant. Use the following context to answer the user's question.\n\nContext: {context} [/INST]",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

    def get_huggingface_llm(self):
        """Initialize and return a HuggingFacePipeline LLM."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=self.max_length,
            truncation=True,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe)

    def get_rag_chain(self, collection_name: str = None):
        """
        Create and return a RAG chain for the specified collection.

        Args:
            collection_name: Name of the Chroma collection (e.g., 'anime', 'tech')

        Returns:
            RAG chain for processing queries
        """
        llm = self.get_huggingface_llm()
        retriever = self.chroma_service.get_retriever(
            collection_name=collection_name, search_kwargs={"k": 2}
        )
        create_history_aware_retriever = history_aware_retriever(
            llm, retriever, self.contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(llm, self.qa_prompt)
        rag_chain = create_retrieval_chain(
            create_history_aware_retriever, question_answer_chain
        )
        return rag_chain
