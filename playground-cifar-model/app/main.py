"""
Backend FastAPI du playground CIFAR-10.

Sert le frontend statique (web/) et expose :
  - GET  /healthz          -> sonde de sante (utilisee par le deploiement srv)
  - GET  /api/game/round   -> une manche du jeu : 3 images CINIC-10 + vraie classe
  - POST /api/predict      -> classe une image (jeu OU upload utilisateur)

Le modele (EfficientNetB5, ~97.9 %) est charge paresseusement au premier appel.
"""
from __future__ import annotations

import io
import os
import mimetypes

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from . import model as model_mod
from . import data as data_mod
from . import leaderboard as lb

app = FastAPI(title="Playground CIFAR-10", docs_url=None, redoc_url=None)

WEB_DIR = os.environ.get("WEB_DIR", "web")

# certains conteneurs ne connaissent pas .webp -> on l'enregistre pour StaticFiles
mimetypes.add_type("image/webp", ".webp")


@app.middleware("http")
async def no_cache_text_assets(request, call_next):
    """Empeche la mise en cache (navigateur + CDN) du HTML/CSS/JS : les
    changements de design sont visibles immediatement, sans cache obsolete."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".css", ".js")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/healthz")
def healthz():
    """Sonde de sante : le service repond toujours 200 s'il est demarre."""
    return {
        "status": "ok",
        "model_loaded": model_mod.model_available(),
        "samples_loaded": data_mod.samples_available(),
    }


@app.get("/api/game/round")
def game_round(n: int = 3):
    """Retourne n images CINIC-10 (jamais vues a l'entrainement) + leur classe."""
    n = max(1, min(n, 20))
    images = data_mod.random_round(n)
    if not images:
        raise HTTPException(
            status_code=503,
            detail="Aucune image d'exemple. Lancez prepare_data.py.",
        )
    return {"images": images}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Classe une image fournie (jeu ou upload). Renvoie top-3 + confiance."""
    if not model_mod.model_available():
        raise HTTPException(
            status_code=503,
            detail="Modele non disponible sur le serveur (model/transfer_model.keras).",
        )
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((32, 32))
    except Exception:
        raise HTTPException(status_code=400, detail="Image invalide.")
    arr = np.array(img, dtype="float32")
    try:
        result = model_mod.predict(arr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return JSONResponse(result)


@app.get("/api/leaderboard")
def get_leaderboard():
    """Classement complet par jeu (le front scrolle au-dela du top 10)."""
    return lb.all_top()


class ScoreIn(BaseModel):
    game: str
    name: str = "Anonyme"
    score: int | None = None
    time: float | None = None
    category: str | None = None
    image: str | None = None


@app.post("/api/score")
def post_score(s: ScoreIn):
    """Enregistre un score (duel) ou un temps+dessin (picto)."""
    try:
        return lb.submit(s.game, s.name, score=s.score, time=s.time,
                         category=s.category, image=s.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/leaderboard")
def reset_leaderboard(game: str | None = None, name: str | None = None,
                      purge: str | None = None,
                      x_reset_token: str | None = Header(default=None)):
    """Vide un classement (ou tous), supprime un pseudo precis (?name=), ou
    (?purge=cheat) nettoie scores forges + dessins couleur unie et restaure
    l'historique. Protege par le token LB_RESET_TOKEN."""
    token = os.environ.get("LB_RESET_TOKEN")
    if not token or x_reset_token != token:
        raise HTTPException(status_code=403, detail="token invalide")
    if purge == "cheat":
        return {"ok": True, "purge": lb.purge_cheaters()}
    lb.reset(game, name)
    return {"ok": True, "reset": game or "all", "name": name}


# Frontend statique monte en DERNIER pour ne pas masquer les routes /api et /healthz.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
