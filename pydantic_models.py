from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    Mixtral_v0_1 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    All_mini_l6_v2 = "sentence-transformers/all-MiniLM-L6-v2"
    DeepSeek_R1_Distill_Qwen_32B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.Mixtral_v0_1)
    collection_name: str = Field(default=None)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int


class StructuredChunkType(str, Enum):
    HEADING = "heading"
    BULLET = "bullet"
    PARAGRAPH = "paragraph"


class StructuredChunk(BaseModel):
    type: StructuredChunkType
    content: str
