from fastapi import APIRouter

from app.api.schema.general.ping import PongResponse


router = APIRouter()


@router.get("/ping", response_model=PongResponse)
def ping():
    return PongResponse()
