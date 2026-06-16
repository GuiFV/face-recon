.PHONY: setup test lint fmt up down logs run

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

# Install the heavy CV stack too (needed to run the pipeline against the camera).
setup-cv:
	$(PIP) install -e ".[cv,dev]"

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check src tests

fmt:
	$(VENV)/bin/ruff format src tests

run:
	$(VENV)/bin/uvicorn face_recon.api.app:create_app --factory --reload --port 8000

up:
	docker compose -f deploy/docker-compose.yml up -d --build

down:
	docker compose -f deploy/docker-compose.yml down

logs:
	docker compose -f deploy/docker-compose.yml logs -f decision-service
