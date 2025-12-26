# ========================================
# HUGGING FACE SPACES DOCKERFILE
# Optimized for SmartDocAI ML Backend
# ========================================

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/backend/app/

# Create persistent data directories
# HF Spaces mounts /data as persistent volume
RUN mkdir -p /data/uploads /data/index /data/.cache && \
    chmod -R 777 /data

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/transformers \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    HUGGINGFACE_SPACES=true \
    SECRET_KEY=hf-spaces-secret-key-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6 \
    DATABASE_URL=sqlite:////data/smartdocai.db

# HF Spaces exposes port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run FastAPI with uvicorn
CMD uvicorn backend.app.main:app --host 0.0.0.0 --port 7860 --workers 1