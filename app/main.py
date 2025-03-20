import uvicorn

from .api.http import app
from .config import config


def main():
    # default는 http서버
    uvicorn.run(app=app, port=config.port, host=config.host)


if __name__ == "__main__":
    main()
