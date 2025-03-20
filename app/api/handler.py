from fastapi import Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse


async def request_validation_exception_handler(_: Request, exc: RequestValidationError):
    detail = str(exc.errors())
    return JSONResponse(
        status_code=400,
        content={"detail": detail},
    )


async def response_validation_exception_handler(_: Request, exc: ResponseValidationError):
    detail = str(exc.errors())
    return JSONResponse(
        status_code=400,
        content={"detail": detail},
    )
