.PHONY: setup lint format test check clean deploy-plan deploy logs upload-data

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

# --- Upload merged training data to S3 ---
# Requires: BUCKET is set (e.g. make upload-data BUCKET=chessgpt-data-chessgpt-l3vmmk)

upload-data:
	@if [ -z "$(BUCKET)" ]; then \
		echo "Error: BUCKET is required. Usage: make upload-data BUCKET=<bucket-name>"; \
		exit 1; \
	fi
	uv run chessgpt-prepare --merge-from-s3 --year 2017 \
		--bucket $(BUCKET) \
		--output-csv data/train_large.csv \
		--fit-tokenizer data/tokenizer_large.json \
		--upload-merged --region us-east-1

# --- AWS Lambda deployment ---
# Requires: terraform apply has been run at least once to create ECR repo

ECR_URL := $(shell cd infra && terraform output -raw ecr_url 2>/dev/null)
AWS_REGION := $(shell cd infra && terraform output -raw aws_region 2>/dev/null || echo us-east-1)

deploy-image:
	@if [ -z "$(ECR_URL)" ]; then \
		echo "Error: ECR_URL is empty. Run 'cd infra && terraform apply' first."; \
		exit 1; \
	fi
	@echo "Building and pushing Lambda container image..."
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_URL)
	docker build --platform linux/arm64 -t chessgpt-lambda -f infra/lambda/Dockerfile .
	docker tag chessgpt-lambda:latest $(ECR_URL):latest
	docker push $(ECR_URL):latest

deploy-plan: deploy-image
	cd infra && terraform plan

deploy: deploy-image
	cd infra && terraform apply

logs:
	@echo "Tailing Lambda logs (Ctrl+C to stop)..."
	@aws logs tail /aws/lambda/chessgpt-download --follow &
	@aws logs tail /aws/lambda/chessgpt-prepare --follow &
	@wait
