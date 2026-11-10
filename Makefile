.PHONY: install test lint integration sweep report site-dev site-build video

install:
	uv venv --python 3.12 .venv
	. .venv/bin/activate && uv pip install -e ".[dev]" && uv pip install --no-deps --no-build-isolation yolox
	cd site && npm install

test:
	. .venv/bin/activate && pytest -q && cd site && npm test

lint:
	. .venv/bin/activate && ruff check src tests

integration:
	. .venv/bin/activate && pytest -q -m integration

sweep:
	. .venv/bin/activate && edgeforge prepare && edgeforge sweep --mode rigorous --run-id $(RUN)

report:
	. .venv/bin/activate && edgeforge report --run-id $(RUN)

site-dev:
	cd site && npm run dev

site-build:
	cd site && npm run build
