from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")

def ping_upload():
    return {"message":"upload route working"}