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
from typing import Dict, List, Optional
import logging

from pydantic_models import ModelName
from langchain_core.retrievers import BaseRetriever


# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaService:
    DEFAULT_COLLECTION_NAME = "default_collection"
    PERSIST_DIRECTORY = "./chroma_db"

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: str = ModelName.All_mini_l6_v2.value,
        chunk_size: int = 500,  # Reduced for smaller documents
        chunk_overlap: int = 100,  # Reduced for smaller documents
    ):
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,  # Preserve metadata for debugging
        )

        self.collection_name = self.DEFAULT_COLLECTION_NAME
        self.vectorstore = Chroma(
            collection_name=self.format_collection_name(self.collection_name),
            persist_directory=self.PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
        )

    def find_similar_documents(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict[str, str]] = None,  # noqa: A002
        where_document: Optional[dict[str, str]] = None,
    ) -> List[Document]:
        documents = self.get_documents()
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k or len(documents),
            filter=filter,
            where_document=where_document,
        )
        filtered_results = [doc for doc, score in results if score < 2.0]
        return filtered_results

    def get_documents(self, file_id: Optional[int] = None) -> List[Dict]:
        """
        Retrieve all document chunks from Chroma that match the given file_id.
        Returns a list of dictionaries containing document content and metadata.
        """
        try:
            results = (
                self.vectorstore._collection.get(
                    where={"file_id": file_id}, include=["documents", "metadatas"]
                )
                if file_id
                else self.vectorstore._collection.get(
                    include=["documents", "metadatas"]
                )
            )

            documents = []
            for i in range(len(results["ids"])):
                documents.append(
                    {
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents for file_id {file_id}: {str(e)}")
            return []

    def format_collection_name(self, name: str) -> str:
        if not name or not name.strip():
            raise HTTPException(
                status_code=400, detail="Collection name cannot be empty or whitespace."
            )

        formatted_name = name.lower().replace(" ", "_")
        formatted_name = re.sub(r"[^a-z0-9_\-.]", "", formatted_name)
        formatted_name = formatted_name.strip("_-.")

        if not formatted_name:
            raise HTTPException(
                status_code=400,
                detail="Collection name is invalid after formatting.",
            )

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
        loader_class = file_loader_map.get(file_extension)

        if not loader_class:
            raise ValueError(f"Unsupported file type: {file_path}")

        try:
            loader = loader_class(file_path)
            print("loader", loader)
            documents = loader.load()
            print("documents", documents)
            print(f"Loaded {len(documents)} documents from {file_path}")
            if not documents:
                raise ValueError(f"No content extracted from {file_path}")

            # Log raw document content for debugging
            for i, doc in enumerate(documents):
                print(f"Document {i} content: {doc.page_content[:500]}...")

            splits = self.text_splitter.split_documents(documents)
            valid_splits = [split for split in splits if split.page_content.strip()]

            # Log split details
            print(
                f"Created {len(splits)} splits, {len(valid_splits)} valid from {file_path}"
            )
            for i, split in enumerate(valid_splits):
                print(f"Split {i} content: {split.page_content[:200]}...")

            if not valid_splits:
                raise ValueError(f"No valid text chunks extracted from {file_path}")
            return valid_splits
        except Exception as e:
            logger.error(f"Failed to load or split document {file_path}: {str(e)}")
            raise ValueError(f"Failed to load or split document {file_path}: {str(e)}")

    def index_document(self, file_path: str, file_id: int) -> bool:
        """
        Load, split, and index a document into Chroma with file_id metadata.
        Returns True on success, False on failure.
        """
        try:
            splits = self.split_document(file_path)
            print(f"Indexing {len(splits)} splits for file_id {file_id}")
            for split in splits:
                split.metadata["file_id"] = file_id
            if splits:
                self.vectorstore.add_documents(splits)
                print(f"Successfully indexed document with file_id {file_id}")
                return True
            else:
                logger.warning(f"No valid splits to index for file_id {file_id}")
                return False
        except Exception as e:
            logger.error(f"Error indexing document with file_id {file_id}: {str(e)}")
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
            logger.error(
                f"Error deleting document with file_id {file_id} from Chroma: {str(e)}"
            )
            return False

    def get_all_collections(self) -> List[str]:
        """
        Retrieve the names of all collections in the Chroma database.
        """
        try:
            collections = self.vectorstore._client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error retrieving collection names: {str(e)}")
            return []

    def get_retriever(self, query: str) -> BaseRetriever:
        return self.vectorstore.as_retriever()
