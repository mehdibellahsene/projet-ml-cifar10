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

import json
import os
import threading

LB_PATH = os.environ.get("LB_PATH", "lbdata/leaderboard.json")
CATEGORIES = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
DUEL_KEEP = 10
PICTO_KEEP = 40      # assez pour couvrir top 10 + 1er de chaque categorie
HISTORY_KEEP = 30    # galerie "mentions honorables" : derniers dessins devines
_lock = threading.Lock()


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
    history = list(reversed(d["history"]))[:24]   # plus recents d'abord
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
            rank = next((i for i, e in enumerate(d["duel"]) if e is entry), None)
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

        entry = {"name": name, "time": t, "category": category}
        if isinstance(image, str) and image.startswith("data:image/") and len(image) < 120000:
            entry["image"] = image
        d["picto"].append(entry)
        d["picto"].sort(key=lambda e: e["time"])
        seen, uniq = set(), []              # meme pseudo + categorie -> garder le plus rapide
        for e in d["picto"]:
            k = (e["name"], e["category"])
            if k in seen:
                continue
            seen.add(k); uniq.append(e)
        d["picto"] = uniq[:PICTO_KEEP]
        d["history"].append(dict(entry))            # historique (ordre d'arrivee)
        d["history"] = d["history"][-HISTORY_KEEP:]
        _save(d)

        rank = next((i for i, e in enumerate(d["picto"]) if e is entry), None)
        cat_best = min((e for e in d["picto"] if e["category"] == category),
                       key=lambda e: e["time"], default=None)
        return {
            "rank": rank if (rank is not None and rank < 10) else None,
            "category_first": cat_best is entry,
        }


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
