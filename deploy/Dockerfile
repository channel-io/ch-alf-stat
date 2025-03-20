FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_LINK_MODE=copy

RUN apt-get update \
  && apt-get install -y automake build-essential protobuf-compiler git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/ ./app

RUN --mount=type=secret,id=GITHUB_TOKEN \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    git config --global url."https://$(cat /run/secrets/GITHUB_TOKEN):x-oauth-basic@github.com/".insteadOf "https://github.com/" \
    && uv sync --frozen --no-install-project --no-dev \
    && git config --global --unset url."https://$(cat /run/secrets/GITHUB_TOKEN):x-oauth-basic@github.com/".insteadOf

ENV HOST="0.0.0.0"

CMD ["uv", "run", "python", "-m", "app.main"]
