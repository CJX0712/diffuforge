PY ?= python
RUFF ?= ruff

.PHONY: help install lint format test demo bench docker clean

help:
	@echo "install  create a venv and install the pinned lockfile"
	@echo "lint     ruff check + ruff format --check (CI gates)"
	@echo "format   apply ruff formatting"
	@echo "test     pytest with coverage"
	@echo "demo     run examples/run_demo.py (writes benchmark.json)"
	@echo "bench    run the full CLI benchmark"
	@echo "docker   build the reproducible image"

install:
	$(PY) -m venv .venv
	.venv/Scripts/python -m pip install --upgrade pip || .venv/bin/python -m pip install --upgrade pip
	.venv/Scripts/python -m pip install -r requirements.lock.txt || .venv/bin/python -m pip install -r requirements.lock.txt

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .

test:
	$(PY) -m pytest -q --cov=diffuforge --cov-report=term-missing

demo:
	$(PY) examples/run_demo.py

bench:
	$(PY) -m diffuforge.cli --out benchmark.json

docker:
	docker build -t diffuforge:0.1.0 .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
