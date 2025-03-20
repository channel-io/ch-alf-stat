from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, ResponseValidationError

from app.config import config

from .handler import request_validation_exception_handler, response_validation_exception_handler
from .route import router


# startup events
def start_up():
    print("Startup event")
    if config.sentry_dsn:
        sentry_sdk.init(
            dsn=config.sentry_dsn,
            traces_sample_rate=config.sentry_traces_sample_rate,
        )


# shutdown events
def shutdown():
    # socket이 close 된 이후 동작할 event list
    print("Graceful shutdown")


# ref) https://fastapi.tiangolo.com/advanced/events/#use-case
@asynccontextmanager
async def lifespan(app: FastAPI):
    start_up()
    yield
    shutdown()


app = FastAPI(
    lifespan=lifespan,
    docs_url=config.docs_url,
    redoc_url=config.redoc_url,
    port=config.port,
    host=config.host,
)

# routes
app.include_router(router)

# exception handlers
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(ResponseValidationError, response_validation_exception_handler)
