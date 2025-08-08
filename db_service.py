import sqlite3
from typing import List, Dict


class DBService:
    def __init__(self, db_name: str = "rag_app.db"):
        self.db_name = db_name
        self._initialize_tables()

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with row factory."""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_tables(self):
        """Initialize the database tables."""
        self._create_application_logs()
        self._create_document_store()

    def _create_application_logs(self):
        """Create the application_logs table if it doesn't exist."""
        with self._get_db_connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     model_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
            )
            conn.commit()

    def _create_document_store(self):
        """Create the document_store table if it doesn't exist."""
        with self._get_db_connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
            )
            conn.commit()

    def insert_application_logs(
        self, session_id: str, user_query: str, model_response: str, model: str
    ):
        """Insert a log entry into application_logs."""
        with self._get_db_connection() as conn:
            conn.execute(
                "INSERT INTO application_logs (session_id, user_query, model_response, model) VALUES (?, ?, ?, ?)",
                (session_id, user_query, model_response, model),
            )
            conn.commit()

    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Retrieve chat history for a given session_id."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_query, model_response FROM application_logs WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
            messages = []
            for row in cursor.fetchall():
                messages.extend(
                    [
                        {"role": "human", "content": row["user_query"]},
                        {"role": "ai", "content": row["model_response"]},
                    ]
                )
            return messages

    def insert_document_record(self, filename: str) -> int:
        """Insert a document record into document_store and return the file_id."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO document_store (filename) VALUES (?)", (filename,)
            )
            file_id = cursor.lastrowid
            conn.commit()
            return file_id

    def delete_document_record(self, file_id: int) -> bool:
        """Delete a document record from document_store by file_id."""
        with self._get_db_connection() as conn:
            conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
            conn.commit()
            return True

    def get_all_documents(self) -> List[Dict]:
        """Retrieve all documents from document_store."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC"
            )
            documents = [dict(doc) for doc in cursor.fetchall()]
            return documents
