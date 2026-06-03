"""
Classement partage (leaderboard) persistant.

- Duel : score (plus haut = mieux), top 10.
- Dessine je devine (picto) : TEMPS en secondes (plus bas = mieux). On conserve
  assez d'entrees pour exposer (1) le top 10 des dessins les plus rapides et
  (2) le 1er de chaque categorie (objet). Chaque entree garde la miniature du dessin.

En production le fichier vit dans un volume Docker (LB_PATH=/srv/lbdata/leaderboard.json)
pour survivre aux redeploiements. Acces serialise (uvicorn mono-worker).
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading

import numpy as np
from PIL import Image

LB_PATH = os.environ.get("LB_PATH", "lbdata/leaderboard.json")
CATEGORIES = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
DUEL_KEEP = 10
PICTO_KEEP = 40        # assez pour couvrir top 10 + 1er de chaque categorie
MAX_DUEL_SCORE = 50000  # au-dela, score forcement forge (anti-triche)
# l'historique "mentions honorables" est illimite : tous les dessins devines
_lock = threading.Lock()


def _is_solid_image(data_url) -> bool:
    """Vrai si le dessin est (quasi) une couleur unie -> triche au pot de peinture.
    On quantifie a 16 niveaux/canal : si une seule couleur couvre >= 95 % de
    l'image, ce n'est pas un dessin."""
    try:
        b64 = str(data_url).split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((32, 32))
        a = np.asarray(img) >> 4
        colors, counts = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
        i = int(counts.argmax())
        if counts[i] / counts.sum() < 0.95:
            return False
        return not bool((colors[i] >= 14).all())   # blanc papier tolere (dessin au trait)
    except Exception:
        return False


def _load() -> dict:
    try:
        with open(LB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data.setdefault("duel", [])
    data.setdefault("picto", [])
    data.setdefault("history", [])
    return data


def _save(d: dict) -> None:
    os.makedirs(os.path.dirname(LB_PATH) or ".", exist_ok=True)
    tmp = LB_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, LB_PATH)


def _clean_name(name) -> str:
    return (str(name or "").strip() or "Anonyme")[:16]


def all_top(n: int = 10) -> dict:
    d = _load()
    # meme pseudo -> ne montrer que son meilleur resultat
    duel, seen = [], set()
    for e in sorted(d["duel"], key=lambda e: -e.get("score", 0)):
        if e.get("name") in seen:
            continue
        seen.add(e.get("name")); duel.append(e)
        if len(duel) >= n:
            break
    picto_sorted = sorted(d["picto"], key=lambda e: e.get("time", 1e9))
    fastest, seen = [], set()
    for e in picto_sorted:
        if e.get("name") in seen:
            continue
        seen.add(e.get("name")); fastest.append(e)
        if len(fastest) >= n:
            break
    by_cat: dict = {}
    for e in picto_sorted:           # deja trie par temps croissant -> 1er vu = plus rapide
        c = e.get("category")
        if c and c not in by_cat:
            by_cat[c] = e
    history = list(reversed(d["history"]))   # plus recents d'abord, sans limite
    return {"duel": duel, "picto_fastest": fastest, "picto_by_cat": by_cat, "picto_history": history}


def submit(game: str, name: str, score=None, time=None, category=None, image=None) -> dict:
    if game not in ("duel", "picto"):
        raise ValueError("jeu inconnu")
    name = _clean_name(name)

    with _lock:
        d = _load()

        if game == "duel":
            try:
                score = max(0, int(score))
            except (TypeError, ValueError):
                raise ValueError("score invalide")
            if score > MAX_DUEL_SCORE:
                raise ValueError("score impossible — arrete de tricher !")
            entry = {"name": name, "score": score}
            d["duel"].append(entry)
            d["duel"].sort(key=lambda e: -e["score"])
            seen, uniq = set(), []          # meme pseudo -> garder le meilleur score
            for e in d["duel"]:
                if e["name"] in seen:
                    continue
                seen.add(e["name"]); uniq.append(e)
            d["duel"] = uniq[:DUEL_KEEP]
            _save(d)
            kept = any(e is entry for e in d["duel"])
            # egalite : rang "competition" = nb de scores strictement meilleurs
            rank = sum(1 for e in d["duel"] if e["score"] > score) if kept else None
            return {"rank": rank if (rank is not None and rank < 10) else None}

        # picto : temps (plus bas = mieux)
        try:
            t = round(float(time), 2)
        except (TypeError, ValueError):
            raise ValueError("temps invalide")
        if t <= 0 or t > 600:
            raise ValueError("temps invalide")
        if category not in CATEGORIES:
            raise ValueError("categorie invalide")
        if not (isinstance(image, str) and image.startswith("data:image/") and len(image) < 120000):
            raise ValueError("dessin manquant")
        if _is_solid_image(image):
            raise ValueError("couleur unie refusee — arrete de tricher !")

        entry = {"name": name, "time": t, "category": category, "image": image}
        d["picto"].append(entry)
        d["picto"].sort(key=lambda e: e["time"])
        seen, uniq = set(), []              # meme pseudo + categorie -> garder le plus rapide
        for e in d["picto"]:
            k = (e["name"], e["category"])
            if k in seen:
                continue
            seen.add(k); uniq.append(e)
        d["picto"] = uniq[:PICTO_KEEP]
        d["history"].append(dict(entry))            # historique (ordre d'arrivee, illimite)
        _save(d)

        kept = any(e is entry for e in d["picto"])
        # egalite : rang "competition" = nb de temps strictement plus rapides
        rank = sum(1 for e in d["picto"] if e["time"] < t) if kept else None
        cat_best = min((e for e in d["picto"] if e["category"] == category),
                       key=lambda e: e["time"], default=None)
        return {
            "rank": rank if (rank is not None and rank < 10) else None,
            "category_first": cat_best is entry,
        }


def purge_cheaters() -> dict:
    """Nettoyage anti-triche + restauration :
    - retire du duel les scores impossibles (> MAX_DUEL_SCORE) ;
    - retire des dessins (classement + historique) les couleurs unies ;
    - reinjecte dans l'historique les dessins encore presents au classement
      picto mais ejectes de l'historique quand il etait limite a 30."""
    with _lock:
        d = _load()
        n_duel = len(d["duel"])
        d["duel"] = [e for e in d["duel"] if e.get("score", 0) <= MAX_DUEL_SCORE]
        n_picto = len(d["picto"])
        d["picto"] = [e for e in d["picto"] if not _is_solid_image(e.get("image", ""))]
        n_hist = len(d["history"])
        d["history"] = [e for e in d["history"] if not _is_solid_image(e.get("image", ""))]
        seen = {(e.get("name"), e.get("time"), e.get("category")) for e in d["history"]}
        restored = [dict(e) for e in d["picto"]
                    if (e.get("name"), e.get("time"), e.get("category")) not in seen]
        d["history"] = restored + d["history"]    # consideres comme les plus anciens
        _save(d)
        return {"duel_retires": n_duel - len(d["duel"]),
                "picto_retires": n_picto - len(d["picto"]),
                "histoire_retires": n_hist - len(d["history"]) + len(restored),
                "dessins_restaures": len(restored)}


def reset(game: str | None = None, name: str | None = None) -> None:
    """Vide un/les classement(s), ou (si name fourni) supprime juste les entrees
    d'un pseudo donne sans toucher au reste."""
    with _lock:
        d = _load()
        lists = ["duel"] if game == "duel" else ["picto", "history"] if game == "picto" else ["duel", "picto", "history"]
        for g in lists:
            if name is not None:
                d[g] = [e for e in d[g] if e.get("name") != name]
            else:
                d[g] = []
        _save(d)
