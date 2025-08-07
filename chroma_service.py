from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List


class ChromaService:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize ChromaService with vector store and text splitter."""
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
        )

    def load_and_split_document(self, file_path: str) -> List[Document]:
        """Load and split a document based on file type."""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".html"):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def index_document(self, file_path: str, file_id: int) -> bool:
        """
        Load, split, and index a document into Chroma with file_id metadata.
        Returns True on success, False on failure.
        """
        try:
            splits = self.load_and_split_document(file_path)
            for split in splits:
                split.metadata["file_id"] = file_id
            self.vectorstore.add_documents(splits)
            return True
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False

    def delete_document(self, file_id: int) -> bool:
        """
        Delete all document chunks with the given file_id from Chroma.
        Returns True on success, False on failure.
        """
        try:
            docs = self.vectorstore.get(where={"file_id": file_id})
            print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
            self.vectorstore._collection.delete(where={"file_id": file_id})
            print(f"Deleted all documents with file_id {file_id}")
            return True
        except Exception as e:
            print(
                f"Error deleting document with file_id {file_id} from Chroma: {str(e)}"
            )
            return False

    def get_retriever(self, search_kwargs: dict = {"k": 2}):
        """Return a retriever for the Chroma vector store."""
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
