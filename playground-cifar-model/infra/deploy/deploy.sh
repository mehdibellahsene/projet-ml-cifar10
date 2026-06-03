#!/usr/bin/env bash
#
# Deploiement de ml-model-detection sur le serveur (Mac mini).
# Synchronise le code, (re)build l'image, redemarre le conteneur, verifie la
# sante puis (re)expose le service via `srv` sur machine-learning.bellahsene.org.
# Idempotent + rollback automatique si la sonde de sante echoue.
#
set -euo pipefail

PUBLIC_HOST="${PUBLIC_HOST:-machine-learning.bellahsene.org}"
PUBLIC_PORT="${PUBLIC_PORT:-8771}"
SRV_BIN="${SRV_BIN:-srv}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-production}"
IMAGE="ml-model-detection"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$APP_DIR"

# --- verrou : un seul deploiement a la fois ---
# Verrou par mkdir (atomique et portable : macOS n'a pas `flock`).
LOCK_DIR="/tmp/deploy-${IMAGE}.lock.d"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Un autre deploiement est en cours, abandon." >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

log() { echo "[deploy] $*"; }

# --- reseau docker partage entre les apps bmr ---
docker network inspect bmr-internal >/dev/null 2>&1 || docker network create bmr-internal

# --- synchronisation du code sur la branche de prod ---
# (sautee quand SKIP_GIT_SYNC=1, p.ex. quand le workflow CI a deja sync.)
if [ -z "${SKIP_GIT_SYNC:-}" ]; then
  REPO_ROOT="$(git -C "$APP_DIR" rev-parse --show-toplevel)"
  log "Synchronisation $DEPLOY_BRANCH dans $REPO_ROOT"
  git -C "$REPO_ROOT" fetch --all --prune
  git -C "$REPO_ROOT" reset --hard "origin/${DEPLOY_BRANCH}"
fi

# --- sauvegarde de l'image actuelle pour rollback ---
if docker image inspect "${IMAGE}:latest" >/dev/null 2>&1; then
  docker tag "${IMAGE}:latest" "${IMAGE}:rollback-prev"
fi

# --- build ---
log "Build de l'image"
PUBLIC_PORT="$PUBLIC_PORT" docker compose build

# --- redemarrage ---
log "Redemarrage du conteneur"
"$SRV_BIN" unexpose "$PUBLIC_HOST" >/dev/null 2>&1 || true
PUBLIC_PORT="$PUBLIC_PORT" docker compose up -d --force-recreate

# --- sonde de sante ---
log "Verification de la sante sur 127.0.0.1:${PUBLIC_PORT}/healthz"
healthy=0
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PUBLIC_PORT}/healthz" >/dev/null 2>&1; then
    healthy=1; break
  fi
  sleep 2
done

if [ "$healthy" -ne 1 ]; then
  log "ECHEC sante -> rollback"
  if docker image inspect "${IMAGE}:rollback-prev" >/dev/null 2>&1; then
    docker tag "${IMAGE}:rollback-prev" "${IMAGE}:latest"
    PUBLIC_PORT="$PUBLIC_PORT" docker compose up -d --force-recreate || true
    "$SRV_BIN" expose "$PUBLIC_PORT" "$PUBLIC_HOST" || true
  fi
  exit 1
fi

# --- exposition publique ---
log "Exposition via srv sur ${PUBLIC_HOST}"
"$SRV_BIN" expose "$PUBLIC_PORT" "$PUBLIC_HOST"

log "Deploiement OK : https://${PUBLIC_HOST}"
