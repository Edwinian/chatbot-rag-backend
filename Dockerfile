# Use Python 3.13 slim base image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with specific versions to reduce size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for SQLite and Chroma with correct permissions
RUN mkdir -p /app/chroma_db && \
    chmod -R 777 /app/chroma_db

# Environment variables (set at runtime or via secrets)
# ENV HUGGINGFACE_TOKEN=<your_token>
# ENV SERPAPI_KEY=<your_key>
# ENV FRONTEND_URL=http://<ec2-public-ip>:3001

# Expose the port
EXPOSE 8000

# Command to run the application with Gunicorn for production
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]