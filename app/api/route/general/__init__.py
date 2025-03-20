from fastapi import APIRouter

from . import health, ping


router = APIRouter(tags=["General resources"])

router.include_router(ping.router)
router.include_router(health.router)
