# Playground CIFAR-10 — application web

Application web qui expose le modèle CIFAR-10 (EfficientNetB5, transfer learning,
**~97,9 %**) à travers trois mini-jeux. Backend **FastAPI**, frontend statique
sans dépendance, déploiement **Docker + `srv`**.

En ligne : **https://machine-learning.bellahsene.org**

## Les trois jeux

| Jeu | Principe | Ce que ça montre |
|-----|----------|------------------|
| **Le Duel : Humain vs Machine** | 10 images, 3 s pour cliquer la bonne classe. L'IA répond en parallèle. Score comparatif à la fin. | Le modèle face à un humain en conditions de rapidité. |
| **Crash Test Pictionary** | Canvas de dessin ; le modèle devine en temps réel à chaque trait. | Les limites du modèle hors distribution (dessins ≠ photos 32×32). |
| **Test Ultime CINIC-10** | Images CINIC-10 (issues d'ImageNet, **jamais vues** à l'entraînement). | La généralisation réelle sur des images inédites. |

## Architecture

```
playground-cifar-model/
├── app/                 # backend FastAPI
│   ├── main.py          #   routes : /healthz, /api/game/round, /api/predict
│   ├── model.py         #   chargement + inférence (image 32x32 brute -> classe)
│   └── data.py          #   banque d'images CINIC-10 étiquetées
├── web/                 # frontend (index.html, style.css, app.js)
├── model/               # transfer_model.keras (48 Mo, EfficientNetB5)
├── data/cinic_samples/  # 250 images CINIC-10 (25/classe) pour les jeux
├── prepare_data.py      # (re)peuple data/cinic_samples depuis CINIC-10
├── Dockerfile           # image runtime (python:3.11-slim + TensorFlow)
├── docker-compose.yml   # service web, réseau bmr-internal
├── Makefile             # build / up / down / deploy
└── infra/deploy/deploy.sh   # build + santé + `srv expose`
```

Le modèle accepte directement une image **32×32×3 brute** (`[0,255]`) : le
redimensionnement vers 456 et le pré-traitement EfficientNet sont intégrés au
graphe sauvegardé. Toute image envoyée (jeu, dessin du canvas, upload) est
ramenée à 32×32 puis classée.

## Lancer en local

```bash
pip install -r requirements.txt
# le modèle doit être présent dans model/transfer_model.keras
uvicorn app.main:app --reload --port 8771
# -> http://127.0.0.1:8771
```

Ou via Docker :

```bash
make build && make up      # http://127.0.0.1:8771
make logs                  # suivre les logs
make down
```

## Déploiement (Mac mini auto-hébergé)

Le déploiement est automatique à chaque push sur la branche **`production`**
(workflow GitHub Actions, runner self-hosted). Le script `infra/deploy/deploy.sh` :

1. synchronise le code sur `origin/production` ;
2. build l'image Docker (tag de rollback conservé) ;
3. redémarre le conteneur via `docker compose` ;
4. vérifie `http://127.0.0.1:8771/healthz` (rollback automatique sinon) ;
5. expose le service avec `srv expose 8771 machine-learning.bellahsene.org`.

Déploiement manuel depuis le serveur :

```bash
make deploy
make deploy-status     # srv status machine-learning.bellahsene.org
```

## Régénérer les images du jeu

```bash
python prepare_data.py --per-class 25     # télécharge CINIC-10 et extrait 25/classe
# ou depuis une archive déjà extraite :
CINIC_DIR=/chemin/CINIC-10 python prepare_data.py
```
