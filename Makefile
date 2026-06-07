.PHONY: setup up down logs

setup:
	docker build -f Dockerfile.sandbox -t resource-agent-sandbox:latest .
	docker compose build

up: setup
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f agent
