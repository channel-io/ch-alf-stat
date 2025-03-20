from fastapi import APIRouter

from app.api.schema.general.health import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()
