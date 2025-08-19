FROM python:3.13-slim

# Install essential dependencies for EasyOCR, OpenCV, and pdf2image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate a virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install typing-extensions from PyPI to avoid metadata issue
RUN pip install --no-cache-dir typing-extensions>=4.10.0

# Install torch (CPU-only version) from PyTorch index
RUN pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install gunicorn explicitly
RUN pip install --no-cache-dir gunicorn==22.0.0

# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for SQLite and Chroma with correct permissions
RUN mkdir -p /app/chroma_db && \
    chmod -R 777 /app/chroma_db

EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]