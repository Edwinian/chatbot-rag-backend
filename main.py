import os
import uuid
import logging
import shutil
import uvicorn
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from pydantic_models import (
    QueryInput,
    QueryResponse,
    DocumentInfo,
    DeleteFileRequest,
)
from langchain_service import LangChainService
from db_service import DBService
from chroma_service import ChromaService

# Set up logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Initialize services
db_service = DBService(db_name="rag_app.db")
chroma_service = ChromaService(persist_directory="./chroma_db")


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


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return db_service.get_all_documents()


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
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
        success = chroma_service.index_document(temp_file_path, file_id)
        if success:
            return {
                "message": f"File {file.filename} has been successfully uploaded and indexed.",
                "file_id": file_id,
            }
        else:
            db_service.delete_document_record(file_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to index {file.filename}."
            )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    langchain_service = LangChainService(
        chroma_service=chroma_service, model_name=query_input.model
    )
    session_id = query_input.session_id or str(uuid.uuid4())
    model_name = query_input.model or langchain_service.model_name

    logging.info(
        f"Session ID: {session_id}, User Query: {query_input.question}, Model: {model_name}"
    )

    chat_history = db_service.get_chat_history(session_id)
    # collection_name = chroma_service.choose_collection(query_input.question)
    rag_chain = langchain_service.get_rag_chain()
    answer = rag_chain.invoke(
        {"input": query_input.question, "chat_history": chat_history}
    )["answer"]
    db_service.insert_application_logs(
        session_id, query_input.question, answer, query_input.model.value
    )
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        stop_signal = False

        async def check_stop():
            nonlocal stop_signal
            while True:
                data = await websocket.receive_json()
                if data.get("action") == "stop":
                    stop_signal = True
                    break

        stop_task = asyncio.create_task(check_stop())

        data = await websocket.receive_json()
        message = data.get("message")
        session_id = data.get("session_id", str(uuid.uuid4()))
        model = data.get("model", None)
        langchain_service = LangChainService(
            chroma_service=chroma_service, model_name=model
        )

        chat_history = db_service.get_chat_history(session_id)
        rag_chain = langchain_service.get_rag_chain()
        answer = rag_chain.invoke({"input": message, "chat_history": chat_history})[
            "answer"
        ]
        chunks = answer.split(". ")

        for i, chunk in enumerate(chunks):
            if stop_signal:
                await websocket.send_json(
                    {"status": "stopped", "session_id": session_id}
                )
                break
            await websocket.send_json(
                {
                    "status": "streaming",
                    "chunk": chunk + (". " if i < len(chunks) - 1 else ""),
                    "session_id": session_id,
                }
            )
            await asyncio.sleep(0.5)

        if not stop_signal:
            await websocket.send_json({"status": "completed", "session_id": session_id})

        stop_task.cancel()
    except Exception as e:
        await websocket.send_json({"error": f"Error: {str(e)}"})
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
