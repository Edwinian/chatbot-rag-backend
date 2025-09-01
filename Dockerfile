FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip wheel

# Install torch and torchvision (CPU-only versions) from PyTorch index
RUN pip install --no-cache-dir \
    torch==2.8.0+cpu \
    torchvision==0.23.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for SQLite and Chroma with correct permissions
RUN mkdir -p /app/chroma_db /app/data && \
    chmod -R 755 /app/chroma_db /app/data

EXPOSE 8000

# Health check for the application
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/get-application-logs || exit 1

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "120", "--log-level", "debug", "main:app", "--bind", "0.0.0.0:8000"]