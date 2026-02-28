# Multilingual E-Commerce RAG Chatbot
# Dockerfile for one-click deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROMA_PATH=/app/data/chroma_db

# Expose ports
# 7860 for Gradio
# 8000 for FastAPI
EXPOSE 7860 8000

# Default command: run Gradio app
CMD ["python", "app.py"]

# Alternative commands:
# For API only: CMD ["python", "api.py"]
# For both: use docker-compose
