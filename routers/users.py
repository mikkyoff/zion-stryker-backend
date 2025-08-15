
from fastapi import APIRouter

router = APIRouter()

@router.get("/me")
def read_user():
    return {"user": "example_user"}  # Placeholder
