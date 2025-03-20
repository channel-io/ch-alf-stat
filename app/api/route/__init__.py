from fastapi import APIRouter

from .general import router as general_router


router = APIRouter()

router.include_router(general_router)
