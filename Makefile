.PHONY: help install install-all server train chat benchmark test lint format clean

help:
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:
	uv sync

install-all:
	uv sync --all-extras

install-training:
	uv sync --extra training

install-benchmarks:
	uv sync --extra benchmarks

install-dev:
	uv sync --extra dev

server:
	uv run uvicorn functiongemma.server:app --host 0.0.0.0 --port 8200 --reload

train:
	uv run python scripts/train-g1.py

chat:
	uv run python scripts/chat-g1.py

benchmark:
	uv run python benchmarks/benchmark-g1-server.py

benchmark-multilingual:
	uv run python benchmarks/benchmark-g1-server-multilingual.py

benchmark-local:
	uv run python benchmarks/benchmark-g1.py

example:
	uv run python examples/chat_client_openai.py

lint:
	uv run ruff check src/ scripts/ benchmarks/ examples/

lint-fix:
	uv run ruff check --fix src/ scripts/ benchmarks/ examples/

format:
	uv run black src/ scripts/ benchmarks/ examples/

format-check:
	uv run black --check src/ scripts/ benchmarks/ examples/

typecheck:
	uv run mypy src/

check: lint format-check typecheck

docker-build:
	docker build -f docker/Dockerfile -t functiongemma-robot-action .

docker-run:
	docker-compose up

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.pyd' -delete

clean-all: clean
	rm -rf .venv/
