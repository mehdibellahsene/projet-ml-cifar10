"""
Classement partage (leaderboard) persistant.

Stocke les meilleurs scores par jeu dans un fichier JSON. En production ce
fichier vit dans un volume Docker (LB_PATH=/srv/lbdata/leaderboard.json) afin
de survivre aux redeploiements. Acces serialise par un verrou (uvicorn mono-worker).
"""
from __future__ import annotations

import json
import os
import threading

LB_PATH = os.environ.get("LB_PATH", "lbdata/leaderboard.json")
GAMES = ("duel", "picto")
KEEP = 50          # nb de scores conserves par jeu
_lock = threading.Lock()


def _load() -> dict:
    try:
        with open(LB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    for g in GAMES:
        data.setdefault(g, [])
    return data


def all_top(n: int = 10) -> dict:
    d = _load()
    return {g: d[g][:n] for g in GAMES}


def _save(d: dict) -> None:
    os.makedirs(os.path.dirname(LB_PATH) or ".", exist_ok=True)
    tmp = LB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, LB_PATH)


def reset(game: str | None = None) -> None:
    """Vide le classement d'un jeu, ou tous si game est None."""
    with _lock:
        d = _load()
        for g in GAMES:
            if game is None or g == game:
                d[g] = []
        _save(d)


def submit(game: str, name: str, score: int, image: str | None = None) -> dict:
    if game not in GAMES:
        raise ValueError("jeu inconnu")
    name = (str(name or "").strip() or "Anonyme")[:16]
    try:
        score = max(0, int(score))
    except (TypeError, ValueError):
        raise ValueError("score invalide")

    with _lock:
        d = _load()
        entry = {"name": name, "score": score}
        # miniature du dessin (jeu picto uniquement) : PNG en data URL, taille bornee
        if game == "picto" and isinstance(image, str) and image.startswith("data:image/") and len(image) < 120000:
            entry["image"] = image
        d[game].append(entry)
        d[game].sort(key=lambda e: -e["score"])
        d[game] = d[game][:KEEP]
        os.makedirs(os.path.dirname(LB_PATH) or ".", exist_ok=True)
        tmp = LB_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, LB_PATH)
        rank = next((i for i, e in enumerate(d[game]) if e is entry), None)

    return {"top": d[game][:10], "rank": rank if (rank is not None and rank < 10) else None}
