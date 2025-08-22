# Chatbot RAG FastAPI Application 
This is a Retrieval-Augmented Generation (RAG) chatbot application built with FastAPI, LangChain, and ChromaDB. It features a user-friendly chat window for real-time interaction with a conversational AI, powered by HuggingFace models, and supports queries enriched with document context and optional web search integration. The application also includes a dedicated page for uploading documents (PDF, DOCX, TXT, images) to be embedded into ChromaDB for retrieval, with automatic PII redaction for enhanced privacy. SQLite is used for storing chat logs and document metadata, ensuring persistence and traceability.


# Create .env File
- Create a .env file in the project root with following keys (values provided separately):
HUGGINGFACE_TOKEN=
SERPAPI_KEY=

# Local setup
- Install Python
- python -m venv venv
- source venv/bin/activate (Windows: venv\Scripts\activate)
- pip install --upgrade pip wheel
- pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu
- pip install -r requirements.txt
- python main.py
- The application will be available at http://localhost:8000

# Docker setup
- Install Docker and docker-compose command
- docker build -t chatbot-rag-app .
- docker run --env-file .env -p 8000:8000 -v $(pwd)/chroma_db:/app/chroma_db -v $(pwd)/data:/app/data --name chatbot-rag-container chatbot-rag-app
- The application will be available at http://localhost:8000


