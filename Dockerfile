FROM python:3.13-slim

# Install essential dependencies for EasyOCR, OpenCV, and pdf2image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate a virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel

# Install torch and torchvision (CPU-only versions) from PyTorch index
RUN pip install --no-cache-dir \
    torch==2.8.0+cpu \
    torchvision==0.20.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create directories for SQLite and Chroma with correct permissions
RUN mkdir -p /app/chroma_db && \
    chmod -R 777 /app/chroma_db

EXPOSE 8000

# Health check for the application
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=1 \
    CMD curl -f http://localhost:8000/get-application-logs || exit 1

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "120", "--log-level", "debug", "main:app", "--bind", "0.0.0.0:8000"]