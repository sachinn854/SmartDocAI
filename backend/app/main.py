# backend/app/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import time
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.database import Base, engine
from app.routes import auth, documents, summarize, ask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the application.
    Ensures database tables are created on first deployment.
    """
    # Startup
    settings = get_settings()
    logger.info("🚀 Starting SmartDocAI API...")
    
    # Ensure directories exist (HF Spaces persistent storage or local)
    try:
        import os
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directories initialized: {settings.UPLOAD_DIR}, {settings.INDEX_DIR}")
    except Exception as e:
        logger.warning(f"⚠️ Directory initialization warning: {e}")
    
    try:
        # Create database tables if they don't exist
        logger.info("Initializing database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        # Don't crash the app - Railway might not have DB ready yet
    
    yield
    
    # Shutdown
    logger.info("Shutting down SmartDocAI API...")


def create_application() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        lifespan=lifespan,
        description="AI-powered document intelligence API with RAG Q&A and summarization",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --------------------------------
    # Rate Limiter Setup
    # --------------------------------
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # --------------------------------
    # Middleware: Request Logging
    # --------------------------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests and their response times."""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"← {request.method} {request.url.path} "
            f"status={response.status_code} time={process_time:.3f}s"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

    # --------------------------------
    # Middleware: Compression (GZip)
    # --------------------------------
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,  # Only compress responses > 1KB
        compresslevel=6     # Balance between speed and compression (1-9)
    )

    # --------------------------------
    # CORS (Environment-aware)
    # --------------------------------
    if settings.ENV == "production":
        # Production: Use specific origins from env
        origins = (
            settings.CORS_ORIGINS.split(",") 
            if settings.CORS_ORIGINS != "*" 
            else ["*"]
        )
        logger.info(f"Production mode - CORS origins: {origins}")
    else:
        # Development: Allow all origins
        origins = ["*"]
        logger.info("Development mode - CORS origins: ['*']")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------
    # Include Routers
    # --------------------------------
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(summarize.router, prefix="/summarize", tags=["Summarization"])
    app.include_router(ask.router, tags=["Question Answering"])

    # --------------------------------
    # Health & Status Endpoints
    # --------------------------------
    @app.get("/")
    def root():
        return {
            "status": "SmartDocAI API running",
            "environment": settings.ENV,
            "version": "1.0.0"
        }
    
    @app.get("/health")
    def health():
        """Health check endpoint for Railway and monitoring."""
        return {
            "status": "healthy",
            "environment": settings.ENV,
            "database": "connected" if settings.DATABASE_URL else "not configured"
        }

    logger.info(f"Application started in {settings.ENV} mode")
    return app


app = create_application()
