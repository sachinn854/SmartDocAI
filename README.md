---
title: DocInsight API
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# DocInsight API 🚀

AI-powered document intelligence backend with:
- 📄 **Document Processing** - PDF, DOCX, images (OCR)
- 🤖 **AI Summarization** - Hierarchical, extractive, abstractive
- 🔍 **RAG Q&A** - FAISS + sentence-transformers
- 🔐 **Authentication** - JWT with user management

## 🌐 API Endpoints

### Base URL
```
https://huggingface.co/spaces/YOUR_USERNAME/docinsight-api
```

### Key Endpoints
- `GET /` - API status
- `GET /health` - Health check
- `POST /auth/register` - User registration
- `POST /auth/login` - Login (returns JWT token)
- `POST /documents/upload` - Upload document
- `GET /documents` - List user's documents
- `POST /summarize/{doc_id}` - Summarize document
- `POST /ask` - Ask questions (RAG)

## 📖 Interactive Documentation
Visit the `/docs` endpoint for interactive Swagger UI:
```
https://huggingface.co/spaces/YOUR_USERNAME/docinsight-api/docs
```

## 🔒 Authentication
Most endpoints require JWT token:
```bash
# 1. Register
curl -X POST https://YOUR-SPACE.hf.space/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}'

# 2. Login to get token
curl -X POST https://YOUR-SPACE.hf.space/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secure123"}'

# 3. Use token in requests
curl -X GET https://YOUR-SPACE.hf.space/documents \
  -H "Authorization: Bearer <your_jwt_token>"
```

## 🛠️ Tech Stack
- **Backend**: FastAPI + Uvicorn
- **ML**: PyTorch + Transformers
- **Embeddings**: Sentence-Transformers
- **Vector DB**: FAISS
- **OCR**: Tesseract
- **Database**: SQLAlchemy + SQLite
- **Auth**: JWT (python-jose)

## 💾 Storage
- Persistent `/data` volume (5GB)
- Uploaded documents: `/data/uploads`
- FAISS index: `/data/index`
- Database: `/data/docinsight.db`

## 🚀 Quick Start

### Example: Upload & Summarize Document

```python
import requests

BASE_URL = "https://YOUR-SPACE.hf.space"

# 1. Register & Login
response = requests.post(f"{BASE_URL}/auth/register", 
    json={"email": "test@example.com", "password": "test123"})

response = requests.post(f"{BASE_URL}/auth/login",
    json={"email": "test@example.com", "password": "test123"})
token = response.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# 2. Upload document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/documents/upload", 
        headers=headers, files=files)
    doc_id = response.json()["document_id"]

# 3. Summarize
response = requests.post(f"{BASE_URL}/summarize/{doc_id}",
    headers=headers,
    json={"summary_type": "extractive", "max_length": 150})
print(response.json()["summary"])

# 4. Ask questions
response = requests.post(f"{BASE_URL}/ask",
    headers=headers,
    json={"question": "What is this document about?"})
print(response.json()["answer"])
```

## 📊 Features

### Document Processing
- PDF text extraction (pdfplumber)
- DOCX processing (python-docx)
- Image OCR (Tesseract)
- Multi-format support

### Summarization
- **Extractive**: TextRank algorithm
- **Abstractive**: DistilBART (lightweight)
- **Hierarchical**: Multi-level summaries
- Configurable length & detail

### Q&A (RAG)
- Semantic search with FAISS
- Sentence-transformers embeddings
- Context-aware answers
- Source attribution

## 🔧 Configuration

Set these secrets in HF Spaces Settings:

```bash
SECRET_KEY=<generate with: openssl rand -hex 32>
DATABASE_URL=sqlite:////data/docinsight.db
CORS_ORIGINS=https://your-frontend.vercel.app
ENV=production
```

## 📈 Resource Usage

- **Memory**: ~500MB base + ~200MB per active request
- **Storage**: ~300MB (dependencies) + user data (max 5GB)
- **CPU**: Optimized for CPU inference (no GPU required)

## 🐛 Troubleshooting

**Issue**: Slow first request  
**Solution**: First request downloads models (~140MB), subsequent requests are fast

**Issue**: "Unauthorized" error  
**Solution**: Ensure JWT token is included in Authorization header

**Issue**: Upload fails  
**Solution**: Check file size (<10MB recommended) and format (PDF/DOCX/images)

## 📝 License
MIT License - See LICENSE file for details

## 🔗 Links
- [GitHub Repository](https://github.com/your-username/DocInsight)
- [API Documentation](https://github.com/your-username/DocInsight/blob/main/docs/API_REFERENCE.md)
- [Deployment Guide](https://github.com/your-username/DocInsight/blob/main/RAILWAY_DEPLOYMENT_GUIDE.md)

---

**Built with ❤️ using FastAPI, Transformers, and Hugging Face Spaces**
