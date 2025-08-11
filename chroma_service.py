import os
import re
from fastapi import HTTPException
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional

from pydantic_models import ModelName


class ChromaService:
    DEFAULT_COLLECTION_NAME = "default_collection"
    PERSIST_DIRECTORY = "./chroma_db"

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: str = ModelName.All_mini_l6_v2.value,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )

        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.vectorstore = Chroma(
            collection_name=self.format_collection_name(
                self.collection_name
            ),  # Create one if not exist
            persist_directory=self.PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
        )

    def format_collection_name(self, name: str) -> str:
        """
        Format collection name to comply with Chroma naming rules:
        - Alphanumeric, underscores, hyphens, periods only.
        - No leading/trailing underscores, hyphens, or periods.
        - No spaces or other special characters.
        - Maximum length of 63 characters.
        - Non-empty.
        """
        if not name or not name.strip():
            raise HTTPException(
                status_code=400, detail="Collection name cannot be empty or whitespace."
            )

        # Convert to lowercase
        formatted_name = name.lower()

        # Replace spaces with underscores
        formatted_name = formatted_name.replace(" ", "_")

        # Keep only alphanumeric, underscores, hyphens, and periods
        formatted_name = re.sub(r"[^a-z0-9_\-.]", "", formatted_name)

        # Remove leading/trailing underscores, hyphens, or periods
        formatted_name = formatted_name.strip("_-.")

        # Check if the resulting name is empty after cleaning
        if not formatted_name:
            raise HTTPException(
                status_code=400,
                detail="Collection name is invalid after formatting (e.g., contains only special characters).",
            )

        # Enforce maximum length
        MAX_COLLECTION_NAME_LENGTH = 63
        if len(formatted_name) > MAX_COLLECTION_NAME_LENGTH:
            formatted_name = formatted_name[:MAX_COLLECTION_NAME_LENGTH].rstrip("_-.")

        return formatted_name

    def split_document(self, file_path: str) -> List[Document]:
        """Load and split a document based on file type."""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_loader_map = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".html": UnstructuredHTMLLoader,
        }
        loader_class = file_loader_map.get(file_extension, None)

        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_path}")

        loader = loader_class(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def index_document(self, file_path: str, file_id: int) -> bool:
        """
        Load, split, and index a document into Chroma with file_id metadata.
        Returns True on success, False on failure.
        """
        try:
            splits = self.split_document(file_path)
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

    def get_retriever(
        self,
        search_kwargs: dict = {"k": 2},
    ):
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_all_collections(self) -> List[str]:
        """
        Retrieve the names of all collections in the Chroma database.
        Returns a list of collection names.
        """
        try:
            collections = self.vectorstore._client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error retrieving collection names: {str(e)}")
            return []
