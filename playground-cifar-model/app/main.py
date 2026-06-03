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

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import model as model_mod
from . import data as data_mod

app = FastAPI(title="Playground CIFAR-10", docs_url=None, redoc_url=None)

WEB_DIR = os.environ.get("WEB_DIR", "web")


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


# Frontend statique monte en DERNIER pour ne pas masquer les routes /api et /healthz.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
