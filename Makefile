.PHONY: setup test lint train deploy mlflow-up

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev,training]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ pipelines/
	ruff format --check src/ pipelines/

format:
	ruff format src/ pipelines/

train:
	python -m pipelines.training_pipeline \
		--model $(or $(MODEL),xgboost) \
		--config $(or $(CONFIG),configs/model_config.yaml)

mlflow-up:
	docker compose -f docker/docker-compose.yml up -d mlflow

mlflow-down:
	docker compose -f docker/docker-compose.yml down

infra-apply:
	cd terraform/environments/$(ENV) && terraform init && terraform apply

deploy:
	kubectl apply -k k8s/overlays/$(ENV)

pipeline-run:
	python -c "from pipelines.training_pipeline import training_pipeline; training_pipeline()"
