"""
Chargement du modèle CIFAR-10 (EfficientNetB5 transfer learning, ~97.9 %) et
inférence. Le modèle accepte directement des images 32x32x3 brutes [0,255] :
le redimensionnement (456) et le pré-traitement EfficientNet sont intégrés au
graphe sauvegardé. On lui passe donc simplement une image 32x32.
"""
from __future__ import annotations

import os
import functools

import numpy as np

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
CLASS_NAMES_FR = {
    "airplane": "avion", "automobile": "automobile", "bird": "oiseau",
    "cat": "chat", "deer": "cerf", "dog": "chien", "frog": "grenouille",
    "horse": "cheval", "ship": "bateau", "truck": "camion",
}

# Le fichier .keras (~120 Mo) n'est PAS versionné : il est placé sur la machine
# hôte (Mac mini) et monté dans le conteneur. Chemin configurable.
MODEL_PATH = os.environ.get("MODEL_PATH", "model/transfer_model.keras")


@functools.lru_cache(maxsize=1)
def _load():
    """Charge le modèle une seule fois (lazy + cache)."""
    import keras  # import tardif : accélère le démarrage si le modèle est absent
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Placez transfer_model.keras "
            "dans le dossier model/ (voir README)."
        )
    return keras.models.load_model(MODEL_PATH)


def model_available() -> bool:
    return os.path.exists(MODEL_PATH)


def predict(image_32: np.ndarray, top_k: int = 3) -> dict:
    """
    Prédit la classe d'une image 32x32x3 (uint8/float, valeurs 0-255).
    Retourne la meilleure classe + les top_k probabilités.
    """
    model = _load()
    x = np.asarray(image_32, dtype="float32")
    if x.ndim == 3:
        x = x[None, ...]  # (1, 32, 32, 3)
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = [{"label": CLASS_NAMES[int(i)],
            "label_fr": CLASS_NAMES_FR[CLASS_NAMES[int(i)]],
            "confidence": float(probs[int(i)])} for i in order[:top_k]]
    best = order[0]
    return {
        "label": CLASS_NAMES[int(best)],
        "label_fr": CLASS_NAMES_FR[CLASS_NAMES[int(best)]],
        "confidence": float(probs[int(best)]),
        "top": top,
    }
