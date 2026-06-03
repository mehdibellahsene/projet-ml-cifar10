"""
Peuple data/cinic_samples/<classe>/ avec quelques images CINIC-10 pour le jeu.

CINIC-10 = images 32x32 derivees d'ImageNet, JAMAIS vues par le modele a
l'entrainement (le modele a appris sur CIFAR-10). C'est ce qui rend le jeu
honnete : on teste la generalisation sur des images inedites.

Usage :
    python prepare_data.py                 # telecharge CINIC-10 et extrait N/classe
    python prepare_data.py --per-class 30
    CINIC_DIR=/chemin/CINIC-10 python prepare_data.py   # depuis un dossier deja extrait

Les images extraites sont minuscules (~1-2 Ko) : on en commite ~20-30 par classe
(total < 1 Mo) pour que le jeu fonctionne sans rien telecharger en production.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import urllib.request

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

CINIC_URL = ("https://datashare.ed.ac.uk/bitstream/handle/10283/3192/"
             "CINIC-10.tar.gz?sequence=4&isAllowed=y")
OUT_DIR = os.environ.get("CINIC_SAMPLES_DIR", "data/cinic_samples")


def _ensure_dirs():
    for c in CLASS_NAMES:
        os.makedirs(os.path.join(OUT_DIR, c), exist_ok=True)


def from_extracted_dir(cinic_dir: str, per_class: int):
    """Copie N images/classe depuis un CINIC-10 deja extrait (test/ de preference)."""
    from PIL import Image
    _ensure_dirs()
    total = 0
    for split in ("test", "valid", "train"):
        base = os.path.join(cinic_dir, split)
        if not os.path.isdir(base):
            continue
        for c in CLASS_NAMES:
            src = os.path.join(base, c)
            if not os.path.isdir(src):
                continue
            existing = len(os.listdir(os.path.join(OUT_DIR, c)))
            need = per_class - existing
            if need <= 0:
                continue
            # on saute les images issues de cifar (prefixe "cifar10-") : on veut de l'inedit
            files = [f for f in sorted(os.listdir(src))
                     if f.lower().endswith(".png") and not f.startswith("cifar10-")]
            for f in files[:need]:
                img = Image.open(os.path.join(src, f)).convert("RGB").resize((32, 32))
                img.save(os.path.join(OUT_DIR, c, f"{c}_{existing}.png"))
                existing += 1
                total += 1
    print(f"{total} images copiees dans {OUT_DIR}")


def from_download(per_class: int):
    """Telecharge l'archive CINIC-10 et extrait N images/classe a la volee (stream)."""
    from PIL import Image
    _ensure_dirs()
    counts = {c: 0 for c in CLASS_NAMES}
    target = per_class * len(CLASS_NAMES)
    print(f"Telechargement CINIC-10 (~660 Mo) puis extraction de {per_class}/classe...")
    with urllib.request.urlopen(CINIC_URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if sum(counts.values()) >= target:
                    break
                if not member.isfile() or not member.name.endswith(".png"):
                    continue
                parts = member.name.split("/")
                if len(parts) < 3:
                    continue
                cls, fname = parts[-2], parts[-1]
                if cls not in counts or counts[cls] >= per_class:
                    continue
                if fname.startswith("cifar10-"):
                    continue  # on garde uniquement l'inedit (ImageNet), pas le cifar
                f = tar.extractfile(member)
                if f is None:
                    continue
                img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((32, 32))
                img.save(os.path.join(OUT_DIR, cls, f"{cls}_{counts[cls]}.png"))
                counts[cls] += 1
                if sum(counts.values()) % 20 == 0:
                    print(f"  {sum(counts.values())}/{target}...")
    print(f"Termine : {sum(counts.values())} images dans {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    args = ap.parse_args()

    cinic_dir = os.environ.get("CINIC_DIR")
    if cinic_dir and os.path.isdir(cinic_dir):
        from_extracted_dir(cinic_dir, args.per_class)
    else:
        from_download(args.per_class)


if __name__ == "__main__":
    sys.exit(main())
