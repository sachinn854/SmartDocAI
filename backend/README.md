---
title: SmartDocAI Backend API
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# SmartDocAI Backend API

A FastAPI-based document processing and analysis backend with ML capabilities.

## Features

- 📄 Document processing (PDF, DOCX, images)
- 🔍 OCR with Tesseract
- 🤖 ML-powered document analysis
- 🔐 Authentication & user management
- 📊 Vector embeddings with FAISS
- 🚀 FastAPI with automatic OpenAPI docs

## API Documentation

Once deployed, visit:
- **Swagger UI**: `https://your-space-name.hf.space/docs`
- **ReDoc**: `https://your-space-name.hf.space/redoc`

## Health Check

- **Health endpoint**: `https://your-space-name.hf.space/health`

## Environment

This application is optimized for Hugging Face Spaces with:
- Port: 7860 (fixed for HF Spaces)
- Persistent storage: `/data` volume
- CPU-optimized PyTorch
- Tesseract OCR support