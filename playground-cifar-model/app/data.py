"""
Gestion des images d'exemple CINIC-10 pour le jeu.

Les images (32x32) sont rangées dans data/cinic_samples/<classe>/*.png.
Ce module les indexe et fournit des tirages aleatoires pour le jeu.
Utiliser prepare_data.py pour telecharger/peupler ce dossier.
"""
from __future__ import annotations

import os
import io
import base64
import random

import numpy as np
from PIL import Image

from .model import CLASS_NAMES

SAMPLES_DIR = os.environ.get("CINIC_SAMPLES_DIR", "data/cinic_samples")


def _index() -> list[tuple[str, str]]:
    """Liste (chemin, classe) de toutes les images d'exemple disponibles."""
    items = []
    for cls in CLASS_NAMES:
        d = os.path.join(SAMPLES_DIR, cls)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    items.append((os.path.join(d, fn), cls))
    return items


def samples_available() -> bool:
    return len(_index()) > 0


def _to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def load_image_array(path: str) -> np.ndarray:
    """Charge une image et la ramene en 32x32x3 (valeurs 0-255)."""
    img = Image.open(path).convert("RGB").resize((32, 32))
    return np.array(img, dtype="float32")


def random_round(n: int = 3) -> list[dict]:
    """Tire n images aleatoires pour une manche du jeu (image + classe reelle)."""
    items = _index()
    if not items:
        return []
    chosen = random.sample(items, min(n, len(items)))
    out = []
    for i, (path, cls) in enumerate(chosen):
        out.append({"id": f"{i}:{path}", "image": _to_b64(path), "true_label": cls})
    return out
