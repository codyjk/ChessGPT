.PHONY: setup lint format test check clean

# Default target: full project setup
setup:
	uv sync --all-extras
	@echo ""
	@echo "Run 'source .venv/bin/activate' to put chessgpt-* commands on PATH."

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest

# Lint + format check + tests (CI-style)
check: lint
	uv run ruff format --check src/ tests/
	uv run pytest

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
