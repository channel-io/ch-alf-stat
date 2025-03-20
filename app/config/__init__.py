import os

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    port: int = 8088
    host: str = "127.0.0.1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    sentry_dsn: str = ""
    sentry_traces_sample_rate: float = 1.0


config = Config()

stage = os.environ.get("STAGE", "development")

print("STAGE: {}".format(stage))

# prevent docs & redoc
if stage == "production":
    config.docs_url = None
    config.redoc_url = None
