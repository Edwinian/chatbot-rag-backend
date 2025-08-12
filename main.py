import os
import uuid
import logging
import shutil
from fastapi.websockets import WebSocketState
from typing import Optional
import uvicorn
import asyncio
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic_models import (
    ModelName,
    QueryInput,
    QueryResponse,
    DocumentInfo,
    DeleteFileRequest,
)
from langchain_service import LangChainService
from db_service import DBService
from chroma_service import ChromaService
from dotenv import load_dotenv

load_dotenv()  # Loads the .env file

# Set up logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL")],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize services
db_service = DBService()
chroma_service = ChromaService()
langchain_service = LangChainService()


@app.get("/find-docs")
def get_documents(file_id: Optional[int] = None):
    documents = chroma_service.get_documents(file_id)

    if not documents:
        raise HTTPException(
            status_code=404, detail=f"No documents found with file_id {file_id}"
        )

    return documents


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = chroma_service.delete_document(request.file_id)
    if chroma_delete_success:
        db_delete_success = db_service.delete_document_record(request.file_id)
        if db_delete_success:
            return {
                "message": f"Successfully deleted document with file_id {request.file_id} from the system."
            }
        else:
            return {
                "error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."
            }
    else:
        return {
            "error": f"Failed to delete document with file_id {request.file_id} from Chroma."
        }


@app.post("/retrieve-content", response_model=str)
def retrieve_content(query_input: QueryInput):
    return langchain_service.retrieve_content(query_input.question)


@app.get("/list-collections", response_model=list[str])
def list_collections():
    return chroma_service.get_all_collections()


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return db_service.get_all_documents()


@app.post("/upload-doc")
async def upload_doc(
    file: UploadFile = File(...), collection_name: Optional[str] = None
):
    chroma_service = ChromaService(collection_name=collection_name)
    allowed_extensions = [".pdf", ".docx", ".html"]
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}",
        )
    temp_file_path = f"temp_{file.filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = db_service.insert_document_record(file.filename)
        print("file_id", file_id)
        success = chroma_service.index_document(temp_file_path, file_id)
        if success:
            logging.info(
                f"File {file.filename} uploaded and indexed with file_id {file_id}"
            )
            return {
                "message": f"File {file.filename} has been successfully uploaded and indexed.",
                "file_id": file_id,
            }
        else:
            db_service.delete_document_record(file_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to index {file.filename}. The document may be empty, contain no extractable text, or have unsupported formatting.",
            )
    except Exception as e:
        db_service.delete_document_record(file_id)
        logging.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing {file.filename}: {str(e)}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    collection_name = query_input.collection_name
    langchain_service = LangChainService(
        model_name=query_input.model, collection_name=collection_name
    )
    session_id = query_input.session_id or str(uuid.uuid4())
    model_name = query_input.model or langchain_service.model_name
    logging.info(
        f"Session ID: {session_id}, User Query: {query_input.question}, Model: {model_name}, Collection: {collection_name}"
    )

    answer = langchain_service.get_model_answer(
        session_id=session_id, query=query_input.question
    )
    db_service.insert_application_logs(
        session_id=session_id,
        user_query=query_input.question,
        model_response=answer,
        model=query_input.model.value,
    )
    logging.info(f"Session ID: {session_id}, Model Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    async def close_websocket(websocket: WebSocket, session_id: Optional[str] = None):
        await websocket.send_json({"status": "closed", "session_id": session_id})
        await websocket.close()
        db_service.delete_application_logs(session_id)

    session_id = str(uuid.uuid4())

    try:
        # Idle timeout (5 minutes)
        idle_timeout = 300  # seconds
        #  manages the timeout internally
        last_active = asyncio.get_event_loop().time()

        while True:
            # Receive client message with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=idle_timeout
                )
                last_active = asyncio.get_event_loop().time()  # Reset activity timer
            except asyncio.TimeoutError:
                # Close idle connection
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(
                        {
                            "status": "timeout",
                            "message": "Connection closed due to inactivity",
                        }
                    )
                    await websocket.close()
                return
            except WebSocketDisconnect:
                break

            # Handle client actions
            action = data.get("action", "open")
            session_id = data.get("session_id", None) or session_id

            if action == "close":
                # Client-initiated closure
                await close_websocket(websocket=websocket, session_id=session_id)
                return

            if action == "stop":
                # Stop current interaction
                await websocket.send_json(
                    {"status": "stopped", "session_id": session_id}
                )
                continue

            # Process chat message
            message = data.get("message")
            if not message:
                await websocket.send_json({"error": "No message provided"})
                continue

            model = data.get("model", ModelName.Mixtral_v0_1.value)
            collection_name = data.get("collection_name", None)

            # Initialize services and generate answer
            langchain_service = LangChainService(
                model_name=model, collection_name=collection_name
            )
            answer = langchain_service.get_model_answer(
                session_id=session_id,
                query=message,
            )

            # Log interaction
            db_service.insert_application_logs(
                session_id=session_id,
                user_query=message,
                model_response=answer,
                model=model,
            )

            # Stream answer in chunks
            chunks = answer.split(". ")
            for i, chunk in enumerate(chunks):
                if websocket.client_state != WebSocketState.CONNECTED:
                    break
                # Check for stop or close signal
                try:
                    stop_data = await asyncio.wait_for(
                        websocket.receive_json(), timeout=0.1
                    )
                    stop_data_action = stop_data.get("action", "open")

                    if stop_data_action == "stop":
                        await websocket.send_json(
                            {"status": "stopped", "session_id": session_id}
                        )
                        break

                    if stop_data_action == "close":
                        await close_websocket(
                            websocket=websocket, session_id=session_id
                        )
                        return
                except asyncio.TimeoutError:
                    pass  # No stop/close message, continue streaming

                is_last_chunk = i == len(chunks) - 1
                await websocket.send_json(
                    {
                        "status": "streaming",
                        "chunk": chunk + (". " if not is_last_chunk else ""),
                        "session_id": session_id,
                        "is_last_chunk": is_last_chunk,
                    }
                )
                await asyncio.sleep(0.5)

            # Send completion status
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(
                    {"status": "completed", "session_id": session_id}
                )

    except Exception as e:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"error": f"Error: {str(e)}"})
            await close_websocket(websocket=websocket, session_id=session_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
