FROM python:3.13-slim  

# Install only essential dependencies for EasyOCR/OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install typing-extensions from PyPI explicitly to avoid metadata issue
RUN pip install --no-cache-dir typing-extensions>=4.10.0

# Install torch first (specify CPU-only version to save space)
RUN pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for SQLite and Chroma with correct permissions
RUN mkdir -p /app/chroma_db && \
    chmod -R 777 /app/chroma_db

EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]