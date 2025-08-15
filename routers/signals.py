
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_signals():
    return {"signals": []}  # Placeholder
