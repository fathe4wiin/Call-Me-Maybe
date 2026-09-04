.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src/__main__.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
