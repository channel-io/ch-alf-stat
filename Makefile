#
# User commands
#
DC=docker compose -f development/docker-compose.yml
PORT ?= 8088
HOST ?= "localhost"

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development
.PHONY: dev
dev:  ## Run development server
	@uv run uvicorn app.api:app --reload --port ${PORT} --host ${HOST}

.PHONY: setup
setup: ## Setup development dependencies in background
	@uv sync
	@uv run pre-commit install

.PHONY: lint
lint: ## Run linter
	@echo Running lint...
	@uv run ruff check --fix .
	@echo Done

.PHONY: format
format: ## Run formatter
	@echo Running formatter...
	@uv run ruff format .
	@echo Done

.PHONY: test
test:  ## Run tests
	@uv run pytest -v test/ -n 4
