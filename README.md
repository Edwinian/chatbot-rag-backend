# Chatbot RAG FastAPI Application 
This is a Retrieval-Augmented Generation (RAG) chatbot application built with FastAPI, LangChain, and ChromaDB. It features a user-friendly chat window for real-time interaction with a conversational AI, powered by HuggingFace models, and supports queries enriched with document context and optional web search integration. The application also includes a dedicated page for uploading documents (PDF, DOCX, TXT, images) to be embedded into ChromaDB for retrieval, with automatic PII redaction for enhanced privacy. SQLite is used for storing chat logs and document metadata, ensuring persistence and traceability.

# .env file
- Unzip .env.zip on root with password provided separately

# Local setup
- Install Python
- python -m venv venv
- source venv/bin/activate (Windows: venv\Scripts\activate)
- pip install --upgrade pip wheel
- pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
- pip install -r requirements.txt
- python main.py
- The application will be running on port 8000 available at http://localhost:8000

# Docker setup
- Install Docker and docker-compose command
- docker-compose up
- The application will be running on port 8000 available at http://localhost:8000


