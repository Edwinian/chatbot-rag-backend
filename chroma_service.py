import os
import re
import base64
from typing import Dict, List, Optional
from fastapi import HTTPException
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging
from pydantic_models import ModelName
from utils_service import UtilsService

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaService:
    DEFAULT_COLLECTION_NAME = "default_collection"
    PERSIST_DIRECTORY = "./chroma_db"
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]
    FILE_EXTENSIONS = [
        ".pdf",
        ".doc",
        ".docx",
        ".html",
        ".txt",
    ]

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: str = ModelName.All_mini_l6_v2.value,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.utils_service = UtilsService()
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                " ",
                "",
            ],
        )

        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.vectorstore = Chroma(
            collection_name=self.format_collection_name(self.collection_name),
            persist_directory=self.PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
        )

    def find_similar_documents(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict[str, str]] = None,
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

            return [
                {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
                for i in range(len(results["ids"]))
            ]
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

    def get_docx_image_documents(self, file_path: str) -> List[Document]:
        try:
            file_path = self.utils_service.convert_docx_to_pdf(file_path)
            return self.get_pdf_image_documents(file_path)
        except Exception as e:
            print(f"Failed to process PDF {file_path}: {str(e)}")
            return []

    def get_pdf_image_documents(self, file_path: str) -> List[Document]:
        try:
            image_paths = self.utils_service.convert_pdf_to_image(file_path)
            print("PDF image_paths", image_paths)
            all_docs = []

            for image_path in image_paths:
                try:
                    img_docs = self.get_image_documents(image_path)
                    print("img_docs", img_docs)

                    if img_docs:
                        all_docs.extend(img_docs)
                except ValueError as e:
                    logger.error(f"Failed to process image {image_path}: {str(e)}")
                    continue
                finally:
                    if os.path.exists(image_path):
                        os.remove(image_path)

            return all_docs
        except Exception as e:
            print(f"Failed to process PDF {file_path}: {str(e)}")
            return []

    def get_image_documents(self, file_path: str) -> List[Document]:
        try:
            results = self.utils_service.extract_texts_from_image(file_path)
            print("texts from images", results)

            if not results:
                return []

            # Combine all detected text into a single string
            extracted_text = "\n".join(results)

            # Create a Document for the extracted text
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "file_path": file_path,
                    "content_type": "image",
                    "file_id": os.path.basename(file_path),
                    "file_extension": os.path.splitext(file_path)[1].lower(),
                    "source": "easyocr",
                },
            )

            return [doc]

        except Exception as e:
            print(f"Failed to process image {file_path}: {str(e)}")
            raise ValueError(f"Failed to process image {file_path}: {str(e)}")

    def get_image_document(self, file_path: str) -> List[Document]:
        try:
            with open(file_path, "rb") as image_file:
                image_data = image_file.read()

            base64_image = base64.b64encode(image_data).decode("utf-8")

            doc = Document(
                page_content=f"IMAGE_CONTENT:{base64_image}",
                metadata={
                    "file_path": file_path,
                    "content_type": "image",
                    "file_id": os.path.basename(file_path),
                    "file_extension": os.path.splitext(file_path)[1].lower(),
                },
            )
            return [doc]
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {str(e)}")
            raise ValueError(f"Failed to process image {file_path}: {str(e)}")

    def get_split_documents(self, file_path: str) -> List[Document]:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in self.IMAGE_EXTENSIONS:
            return self.get_image_documents(file_path)

        file_loader_map = {
            ".pdf": UnstructuredPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".html": UnstructuredHTMLLoader,
            ".txt": lambda x: [Document(page_content=open(x, "r").read())],
        }
        loader_class = file_loader_map.get(file_extension)

        if not loader_class:
            raise ValueError(f"File loader not found: {file_path}")

        try:
            if file_extension == ".txt":
                documents = loader_class(file_path)
            else:
                loader = loader_class(file_path, mode="elements")
                documents = loader.load()

            # Get documents from images as loader only gets documents from texts
            # Comment out due to performance issues
            # if file_extension == ".pdf":
            #     print("file_extension", file_extension)
            #     img_documents = self.get_pdf_image_documents(file_path)
            #     print(f"{file_extension} img document count", len(img_documents))
            #     documents += img_documents

            # if file_extension == ".docx":
            #     print("file_extension", file_extension)
            #     img_documents = self.get_docx_image_documents(file_path)
            #     print(f"{file_extension} img document count", len(img_documents))
            #     documents += img_documents

            processed_docs = []
            current_content = ""

            for doc in documents:
                content = doc.page_content.strip()
                if len(content) < 10:
                    current_content += " " + content
                else:
                    if current_content:
                        processed_docs.append(
                            Document(
                                page_content=current_content.strip(),
                                metadata=doc.metadata,
                            )
                        )
                        current_content = ""
                    processed_docs.append(doc)

            if current_content:
                processed_docs.append(
                    Document(
                        page_content=current_content.strip(),
                        metadata=documents[-1].metadata,
                    )
                )

            splits = self.text_splitter.split_documents(processed_docs)
            return splits
        except Exception as e:
            logger.error(f"Failed to load or split document {file_path}: {str(e)}")
            raise ValueError(f"Failed to load or split document {file_path}: {str(e)}")

    def get_valid_splits(
        self, splits: List[Document]
    ) -> tuple[List[Document], list[str]]:
        valid_splits = []
        pii_content = []
        presidio_patterns = {
            # Personal Identifiers
            "PERSON": r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b",  # Names (simple pattern)
            "US_SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "NRIC": r"\b[STFG]\d{7}[A-Z]\b",  # Singapore ID
            "PASSPORT": r"\b[A-Z]{1,2}\d{6,9}\b",
            # Contact Information
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            # Financial
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
            "SWIFT_CODE": r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
            # Medical
            "MEDICAL_LICENSE": r"\b[A-Z]{2,3}\d{5,8}\b",
            # Location (simple patterns)
            "ADDRESS": r"\b\d{1,5} [A-Za-z]+(?: [A-Za-z]+){1,3},? [A-Z]{2} \d{5}\b",
            "COORDINATES": r"\b-?\d{1,3}\.\d{4,}, -?\d{1,3}\.\d{4,}\b",
        }
        iam_smart_patterns = {
            # Name Identifiers
            "CHINESE_NAME": r"[\u4e00-\u9fff]{2,4}",  # 2-4 Chinese characters
            "ENGLISH_NAME": r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b",
            # Government IDs
            "HKID": r"\b[A-Z]{1,2}[0-9]{6}\([0-9A]\)\b",  # Official HKID format
            "PASSPORT": r"\b[A-Z]{1,3}\d{6,9}\b",
            # Contact Information
            "PRIMARY_EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "MOBILE_PHONE": r"\b(852)?[ -]?\d{4}[ -]?\d{4}\b",  # HK mobile format
            # Address Information (Hong Kong specific)
            "RESIDENTIAL_ADDRESS": r"\b(Flat|Floor|Room|Unit|Villa)[\sA-Z0-9-#]+,?\s[\w\s]+(Hong Kong|HK|H\.K\.|New Territories|NT|Kowloon|KLN)\b",
            "POSTAL_ADDRESS": r"\b(P\.O\. Box|G\.P\.O\. Box|Post Office Box)\s\d+\b",
            # Financial
            "BANK_ACCOUNT": r"\b\d{10,12}\b",  # Simplified HK bank account
        }
        pii_patterns = {
            **presidio_patterns,
            **iam_smart_patterns,
        }

        for split in splits:
            if split.page_content.startswith("IMAGE_CONTENT:"):
                valid_splits.append(split)
                continue

            try:
                content = split.page_content
                redacted_content = content
                split_pii_detected = False

                # Check for PII patterns
                for pii_type, pattern in pii_patterns.items():
                    matches = re.finditer(pattern, content, re.IGNORECASE)

                    for match in matches:
                        split_pii_detected = True
                        redacted_content = redacted_content.replace(
                            match.group(), f"[REDACTED_{pii_type.upper()}]"
                        )

                # Update the content if PII was found
                if split_pii_detected:
                    split.page_content = redacted_content
                    pii_content.append(redacted_content)
                    logger.info(
                        f"PII detected and redacted in document split: {content[:100]}..."
                    )

                # Basic validation checks
                len_check = split.page_content.strip() and len(split.page_content) >= 5
                embedding = self.embedding_function.embed_query(split.page_content)

                if all([len_check, embedding is not None, any(embedding)]):
                    valid_splits.append(split)

            except Exception as e:
                logger.error(f"Failed to process split: {str(e)}")
                continue

        return valid_splits, pii_content

    def index_document(self, file_path: str, file_id: int) -> dict[str, str]:
        try:
            splits = self.get_split_documents(file_path)
            valid_splits, pii_content = self.get_valid_splits(splits)
            response = {
                "success": "0",
                "error": "",
                "pii_content": "".join(pii_content),
            }

            if not valid_splits:
                response["error"] = "No valid document splits found."
                return response

            pii_detected = len(pii_content) > 0

            for split in valid_splits:
                split.metadata["file_id"] = file_id
                split.metadata = {
                    k: str(v) if v is not None else ""
                    for k, v in split.metadata.items()
                }
                # Add PII detection flag to metadata
                split.metadata["pii_detected"] = str(pii_detected)

            if valid_splits:
                filtered_splits = filter_complex_metadata(valid_splits)
                self.vectorstore.add_documents(filtered_splits)
                response["success"] = "1"

            return response
        except Exception as e:
            logger.error(f"Error indexing document with file_id {file_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index {os.path.basename(file_path)}: {str(e)}",
            )

    def delete_document(self, file_id: int) -> bool:
        try:
            self.vectorstore._collection.delete(where={"file_id": file_id})
            return True
        except Exception as e:
            logger.error(f"Error deleting document with file_id {file_id}: {str(e)}")
            return False

    def get_all_collections(self) -> List[str]:
        try:
            collections = self.vectorstore._client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error retrieving collection names: {str(e)}")
            return []

    def get_retriever(self) -> BaseRetriever:
        return self.vectorstore.as_retriever()
