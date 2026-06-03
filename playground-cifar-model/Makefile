PROJECT     ?= ml-model-detection
PUBLIC_HOST ?= machine-learning.bellahsene.org
PUBLIC_PORT ?= 8771
SRV_BIN     ?= srv

export PUBLIC_HOST PUBLIC_PORT SRV_BIN PROJECT

.PHONY: deploy deploy-status build up down logs ps

## Deploiement complet (build + up + expose via srv) — appele par le runner GitHub Actions
deploy:
	bash infra/deploy/deploy.sh

## Etat de l'exposition publique
deploy-status:
	$(SRV_BIN) show $(PUBLIC_HOST)

build:
	PUBLIC_PORT=$(PUBLIC_PORT) docker compose build

up:
	PUBLIC_PORT=$(PUBLIC_PORT) docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps
