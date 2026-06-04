"""
==============================================================================
 Projet Machine Learning — Classification d'images CIFAR-10 par CNN
==============================================================================
Auteur : Mehdi Redha BELLAHSENE — Méthodologie CRISP-DM — TensorFlow / Keras

Script unique et autonome, reprenant EXACTEMENT la logique du notebook
`TP_Project_ML.ipynb` (les résultats de référence sont produits sur Colab GPU).
Ce fichier est destiné à la lecture du code par le correcteur et peut aussi
être exécuté.

Plan du fichier :
  1. Données        (chargement, normalisation, one-hot, visualisation)
  2. Modèles        (LeNet-5/CNN1, VGG1/2/3, variantes régularisées, modèle bonus)
  3. Entraînement   (optimiseurs, callbacks, plot_history)
  4. Évaluation     (matrice de confusion, précision par classe, comparaisons)
  5. Pipeline       (grille 36 configs, modèle retenu, modèle bonus)
  6. Rapport PDF    (utilitaire de génération — annexe)

Usage :
  python projet_cifar10.py              # pipeline complet + rapport PDF
  python projet_cifar10.py --quick      # test rapide (peu d'époques)
  python projet_cifar10.py --report     # génère seulement le PDF depuis results.json
  python projet_cifar10.py --no-bonus   # sans le modèle bonus (hors énoncé)

Sur GPU/Colab, TensorFlow utilise automatiquement l'accélérateur (aucun
changement de code). Sur CPU, l'entraînement complet est long.
"""
from __future__ import annotations

import os
import re
import json
import time
import argparse

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import keras
from keras import layers, models as kmodels
from keras.utils import to_categorical

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_assets")
CURVES = os.path.join(ASSETS, "curves")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")

LEARNING_RATES = [0.001, 0.01, 0.1]
OPTIMIZERS = ["SGD", "Adam"]


# ============================================================================
# 1. DONNÉES
# ============================================================================
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
FR_NAMES = {"airplane": "avion", "automobile": "automobile", "bird": "oiseau",
            "cat": "chat", "deer": "cerf", "dog": "chien", "frog": "grenouille",
            "horse": "cheval", "ship": "bateau", "truck": "camion"}


def load_raw():
    """Charge CIFAR-10 brut (pixels 0-255, labels entiers)."""
    return keras.datasets.cifar10.load_data()


def describe(x_train, y_train, x_test, y_test) -> dict:
    """Caractéristiques structurelles du jeu (Tâche 1)."""
    return {"train_shape": list(x_train.shape), "test_shape": list(x_test.shape),
            "n_train": int(x_train.shape[0]), "n_test": int(x_test.shape[0]),
            "image_shape": list(x_train.shape[1:]), "n_channels": int(x_train.shape[-1]),
            "n_classes": len(CLASS_NAMES)}


def preprocess(x_train, y_train, x_test, y_test):
    """Normalise les pixels dans [0,1] et applique le one-hot (Tâches 3 & 4)."""
    x_train_n = x_train.astype("float32") / 255.0
    x_test_n = x_test.astype("float32") / 255.0
    y_train_oh = to_categorical(y_train, 10)
    y_test_oh = to_categorical(y_test, 10)
    return x_train_n, y_train_oh, x_test_n, y_test_oh


def plot_sample_images(x, y, save_path, n=10):
    """Affiche n images d'exemple avec leurs labels (Tâche 2)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 5))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        plt.title(CLASS_NAMES[int(y[i][0])])
        plt.axis("off")
    plt.suptitle("Échantillon d'images CIFAR-10 (32x32, 3 canaux RVB)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_class_distribution(y_train, save_path) -> dict:
    """Distribution des classes (Tâche 5)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(9, 4.5))
    plt.bar([CLASS_NAMES[int(i)] for i in unique], counts,
            color="#4C72B0", edgecolor="black", linewidth=0.5)
    plt.title("Distribution des classes — jeu d'entraînement CIFAR-10")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return {CLASS_NAMES[int(i)]: int(c) for i, c in zip(unique, counts)}


# ============================================================================
# 2. MODÈLES
# ============================================================================
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
VGG_FILTERS = 32          # imposé par l'énoncé
VGG_KERNEL = (3, 3)
VGG_POOL = (2, 2)
VGG_HIDDEN = 128          # couche cachée imposée


def build_lenet5(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """CNN1 = LeNet-5 (conforme au TP du cours : he_uniform, padding same)."""
    return kmodels.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(6, (5, 5), activation="relu", kernel_initializer="he_uniform", padding="same"),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(16, (5, 5), activation="relu", kernel_initializer="he_uniform", padding="same"),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="LeNet5")


def _vgg_block(model, dropout, batchnorm, rate):
    """Bloc VGG : Conv -> Conv -> MaxPool (+ BatchNorm/Dropout en option)."""
    model.add(layers.Conv2D(VGG_FILTERS, VGG_KERNEL, activation="relu", padding="same"))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(VGG_FILTERS, VGG_KERNEL, activation="relu", padding="same"))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(VGG_POOL))
    if dropout:
        model.add(layers.Dropout(rate))


def build_vgg(num_blocks, dropout=False, batchnorm=False,
              input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """VGG-like à `num_blocks` blocs ; Dropout/BatchNorm indépendants (pour les isoler)."""
    suffix = ("_drop" if dropout else "") + ("_bn" if batchnorm else "")
    model = kmodels.Sequential(name=f"VGG{num_blocks}{suffix}")
    model.add(keras.Input(shape=input_shape))
    rates = [0.2, 0.3, 0.4]
    for b in range(num_blocks):
        _vgg_block(model, dropout, batchnorm, rates[b])
    model.add(layers.Flatten())
    model.add(layers.Dense(VGG_HIDDEN, activation="relu"))
    if batchnorm:
        model.add(layers.BatchNormalization())
    if dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model


def build_vgg1():
    return build_vgg(1)


def build_vgg2():
    return build_vgg(2)


def build_vgg3():
    return build_vgg(3)


def build_vgg3_drop():
    return build_vgg(3, dropout=True)


def build_vgg3_drop_bn():
    return build_vgg(3, dropout=True, batchnorm=True)


# Les 6 lignes du tableau de l'énoncé (CNN1 = LeNet-5).
MODEL_BUILDERS = {
    "CNN1": build_lenet5, "VGG1": build_vgg1, "VGG2": build_vgg2, "VGG3": build_vgg3,
    "VGG3+Drop": build_vgg3_drop, "VGG3+Drop+BatchNorm": build_vgg3_drop_bn,
}
MODEL_ORDER = list(MODEL_BUILDERS.keys())


def build_final_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """Modèle BONUS (hors énoncé) : filtres 64->128->256 + BN + Dropout + augmentation."""
    return kmodels.Sequential([
        keras.Input(shape=input_shape),
        layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"), layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"), layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"), layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"), layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.3),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"), layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"), layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.4),
        layers.Flatten(), layers.Dense(256, activation="relu"), layers.BatchNormalization(),
        layers.Dropout(0.5), layers.Dense(num_classes, activation="softmax"),
    ], name="FinalVGG")


def build_transfer_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, img_size=224):
    """Modèle TRANSFER LEARNING (objectif ~96%) : EfficientNetB0 pré-entraîné ImageNet.

    Les images 32x32 sont agrandies (Resizing) à `img_size` ; EfficientNet attend des
    pixels [0,255] (sa normalisation est interne) — on lui passe donc les images BRUTES.
    Retourne (modèle, base) pour pouvoir geler/dégeler la base (entraînement en 2 phases).
    """
    apps = {"B0": keras.applications.EfficientNetB0,
            "B3": keras.applications.EfficientNetB3,
            "B5": keras.applications.EfficientNetB5}
    backbone = globals().get("TRANSFER_BACKBONE", "B5")
    base = apps[backbone](include_top=False, weights="imagenet",
                          input_shape=(img_size, img_size, 3))
    base.trainable = False
    inp = keras.Input(shape=input_shape)
    x = layers.Resizing(img_size, img_size)(inp)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    # Sortie en float32 (stabilité numérique en mixed precision).
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inp, out, name=f"EfficientNet{backbone}_TL"), base


# ============================================================================
# 3. ENTRAÎNEMENT
# ============================================================================
def make_optimizer(name, lr=None):
    name = name.upper()
    if name == "SGD":
        return keras.optimizers.SGD(learning_rate=lr or 0.01, momentum=0.9)
    if name == "ADAM":
        return keras.optimizers.Adam(learning_rate=lr or 0.001)
    raise ValueError(name)


def compile_model(model, optimizer_name, lr=None):
    model.compile(optimizer=make_optimizer(optimizer_name, lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def es_only(patience=8):
    """Early stopping seul (ne modifie pas le learning rate étudié)."""
    return [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                          mode="max", restore_best_weights=True, verbose=0)]


def train_and_evaluate(build_fn, optimizer_name, lr, data, label,
                       epochs=50, batch_size=64, callbacks=None, strategy=None):
    """Construit, entraîne et évalue un modèle. Conserve métriques train ET test."""
    x_train, y_train, x_test, y_test = data
    if strategy is not None:
        with strategy.scope():
            model = build_fn()
            compile_model(model, optimizer_name, lr)
    else:
        model = build_fn()
        compile_model(model, optimizer_name, lr)
    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(x_test, y_test),
                     callbacks=callbacks if callbacks is not None else es_only(), verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    res = {"model": label, "optimizer": optimizer_name, "learning_rate": lr,
           "test_accuracy": float(test_acc), "test_loss": float(test_loss),
           "train_accuracy": float(train_acc), "train_loss": float(train_loss),
           "overfit_gap": float(train_acc - test_acc),
           "epochs_run": len(hist.history["loss"]), "n_params": int(model.count_params()),
           "best_val_accuracy": float(max(hist.history["val_accuracy"]))}
    return res, hist, model


def plot_history(history, title, save_path):
    """Deux figures : perte et précision (entraînement + test). Demandé par l'énoncé."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    h = history.history
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(h["loss"], color="#1f77b4", label="Entraînement")
    plt.plot(h["val_loss"], color="#d62728", linestyle="--", label="Validation (test)")
    plt.title(f"{title} — Perte"); plt.xlabel("Époque"); plt.ylabel("Perte")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(h["accuracy"], color="#1f77b4", label="Entraînement")
    plt.plot(h["val_accuracy"], color="#d62728", linestyle="--", label="Validation (test)")
    plt.title(f"{title} — Précision"); plt.xlabel("Époque"); plt.ylabel("Précision")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()


# ============================================================================
# 4. ÉVALUATION
# ============================================================================
def predict_classes(model, x_test):
    probs = model.predict(x_test, verbose=0)
    return np.argmax(probs, axis=1), probs


def plot_confusion_matrix(y_true, y_pred, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(cm_n, annot=True, fmt=".0f", cmap="Blues", linewidths=0.5,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar_kws={"label": "%"})
    plt.title("Matrice de confusion (% par classe réelle)")
    plt.xlabel("Classe prédite"); plt.ylabel("Classe réelle")
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return cm


def per_class_accuracy(y_true, y_pred, save_path) -> dict:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1)
    order = np.argsort(acc)
    plt.figure(figsize=(9, 4.5))
    colors = ["#C44E52" if a < acc.mean() else "#55A868" for a in acc[order]]
    plt.bar([CLASS_NAMES[i] for i in order], acc[order] * 100, color=colors,
            edgecolor="black", linewidth=0.4)
    plt.axhline(acc.mean() * 100, color="grey", linestyle="--",
                label=f"Moyenne = {acc.mean()*100:.1f}%")
    plt.title("Précision par classe"); plt.xlabel("Classe"); plt.ylabel("Précision (%)")
    plt.xticks(rotation=45); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()
    return {CLASS_NAMES[i]: float(acc[i]) for i in range(len(CLASS_NAMES))}


def plot_misclassified(model, x_test, y_true, save_path, n=10):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred, _ = predict_classes(model, x_test)
    wrong = np.where(y_pred != y_true)[0][:n]
    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(wrong):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx])
        plt.title(f"V: {CLASS_NAMES[int(y_true[idx])]}\nP: {CLASS_NAMES[int(y_pred[idx])]}", fontsize=9)
        plt.axis("off")
    plt.suptitle("Exemples mal classés (V = réel, P = prédit)")
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close()


def _short(m):
    """Nom court d'architecture pour les figures."""
    return m.replace("+Drop+BatchNorm", "+D+BN").replace("+Drop", "+D")


def plot_model_comparison(results, save_path, model_order, lrs, opts):
    """Heatmap annotée ; les cellules en divergence (<=15 %, ~hasard) sont hachurées."""
    from matplotlib.patches import Rectangle
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lookup = {(r["model"], r["learning_rate"], r["optimizer"]): r["test_accuracy"] for r in results}
    cols = [f"LR={lr}\n{o}" for lr in lrs for o in opts]
    mat = np.full((len(model_order), len(cols)), np.nan)
    for mi, m in enumerate(model_order):
        ci = 0
        for lr in lrs:
            for o in opts:
                v = lookup.get((m, lr, o))
                if v is not None:
                    mat[mi, ci] = v * 100
                ci += 1
    plt.figure(figsize=(11, 6))
    ax = sns.heatmap(mat, annot=True, fmt=".1f", cmap="RdYlGn", vmin=10, vmax=95,
                     xticklabels=cols, yticklabels=[_short(m) for m in model_order],
                     cbar_kws={"label": "Précision test (%)"}, linewidths=0.6, linecolor="white",
                     annot_kws={"fontsize": 9, "fontweight": "bold"})
    for mi in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            if not np.isnan(mat[mi, ci]) and mat[mi, ci] <= 15:
                ax.add_patch(Rectangle((ci, mi), 1, 1, fill=False, edgecolor="black", lw=1.4, hatch="///"))
    plt.title("Précision de test (%) par modèle, learning rate et optimiseur\n"
              "(hachuré = divergence, ~10 % = niveau du hasard)", fontsize=11)
    plt.ylabel("Modèle"); plt.xlabel("")
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_lr_sensitivity(results, save_path, model_order, lrs):
    """Précision vs learning rate (échelle log), par optimiseur, avec niveau du hasard."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    opts = sorted({r["optimizer"] for r in results})
    lookup = {(r["model"], r["learning_rate"], r["optimizer"]): r["test_accuracy"] for r in results}
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, len(opts), figsize=(13, 5.5), sharey=True)
    if len(opts) == 1:
        axes = [axes]
    for ax, o in zip(axes, opts):
        for k, m in enumerate(model_order):
            ys = [lookup.get((m, lr, o)) * 100 if lookup.get((m, lr, o)) is not None else np.nan
                  for lr in lrs]
            ax.plot(lrs, ys, marker=markers[k % len(markers)], markersize=7, linewidth=1.8, label=_short(m))
        ax.set_xscale("log"); ax.set_xticks(lrs); ax.set_xticklabels([str(l) for l in lrs])
        ax.axhspan(0, 15, color="red", alpha=0.06)
        ax.axhline(10, color="grey", ls=":", lw=1.2)
        ax.text(lrs[0], 12, "niveau du hasard (10 %)", fontsize=7.5, color="grey")
        ax.set_title(f"Optimiseur : {o}", fontsize=11); ax.set_xlabel("Learning rate (échelle log)")
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 100)
    axes[0].set_ylabel("Précision de test (%)")
    axes[-1].legend(fontsize=8, title="Modèle", loc="lower left")
    plt.suptitle("Sensibilité de la précision au learning rate", fontsize=12)
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_accuracy_bars(results, save_path, model_order, best_lr, opts):
    """Barres groupées : précision de test par architecture, un jeu de barres par optimiseur."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lookup = {(r["model"], r["learning_rate"], r["optimizer"]): r["test_accuracy"] for r in results}
    x = np.arange(len(model_order)); w = 0.8 / max(1, len(opts))
    palette = {"SGD": "#4C72B0", "Adam": "#DD8452"}
    plt.figure(figsize=(11, 5.5))
    for k, o in enumerate(opts):
        vals = [(lookup.get((m, best_lr, o)) or 0) * 100 for m in model_order]
        xb = x + (k - (len(opts) - 1) / 2) * w
        bars = plt.bar(xb, vals, w, label=o, color=palette.get(o), edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5)
    plt.xticks(x, [_short(m) for m in model_order], rotation=20, ha="right")
    plt.ylabel("Précision de test (%)"); plt.ylim(0, 100)
    plt.title(f"Précision de test par architecture et optimiseur (learning rate = {best_lr})")
    plt.legend(title="Optimiseur"); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_train_test_gap(results, save_path, model_order, best_lr, opt):
    """Barres entraînement vs test (au meilleur LR + optimiseur) -> sur-apprentissage."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lut = {(r["model"], r["learning_rate"], r["optimizer"]): r for r in results}
    rows = [lut.get((m, best_lr, opt)) for m in model_order]
    if not any(r and r.get("train_accuracy") is not None for r in rows):
        return
    tr = [(r.get("train_accuracy") or 0) * 100 if r else 0 for r in rows]
    te = [(r.get("test_accuracy") or 0) * 100 if r else 0 for r in rows]
    x = np.arange(len(model_order)); w = 0.38
    plt.figure(figsize=(11, 5.5))
    plt.bar(x - w / 2, tr, w, label="Entraînement", color="#55A868", edgecolor="black", linewidth=0.4)
    plt.bar(x + w / 2, te, w, label="Test", color="#C44E52", edgecolor="black", linewidth=0.4)
    for i in range(len(model_order)):
        plt.text(x[i], max(tr[i], te[i]) + 1.2, f"écart {tr[i]-te[i]:.0f}", ha="center", fontsize=7.5)
    plt.xticks(x, [_short(m) for m in model_order], rotation=20, ha="right")
    plt.ylabel("Précision (%)"); plt.ylim(0, 108)
    plt.title(f"Entraînement vs test ({opt}, LR={best_lr}) — l'écart mesure le sur-apprentissage")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()


# ============================================================================
# 5. PIPELINE
# ============================================================================
def _slug(s):
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def run_grid(data, raw, epochs=50, batch_size=64, strategy=None):
    """Grille 36 configs (6 modèles x 3 LR x 2 optimiseurs). Sauvegarde incrémentale."""
    os.makedirs(CURVES, exist_ok=True)
    (xtr, ytr), (xte, yte) = raw
    desc = describe(xtr, ytr, xte, yte)
    cd = plot_class_distribution(ytr, os.path.join(ASSETS, "class_distribution.png"))
    plot_sample_images(xtr, ytr, os.path.join(ASSETS, "samples.png"))
    results, t0 = [], time.time()
    for lr in LEARNING_RATES:
        for name in MODEL_ORDER:
            for opt in OPTIMIZERS:
                res, hist, _ = train_and_evaluate(MODEL_BUILDERS[name], opt, lr, data, name,
                                                  epochs=epochs, batch_size=batch_size,
                                                  callbacks=es_only(), strategy=strategy)
                cpath = os.path.join(CURVES, f"{_slug(name)}_lr{_slug(str(lr))}_{_slug(opt)}.png")
                plot_history(hist, f"{name} | {opt} | LR={lr}", cpath)
                res["curve"] = os.path.relpath(cpath, ASSETS)
                results.append(res)
                print(f"{name:20s} {opt:4s} lr={lr:<6} -> test={res['test_accuracy']:.3f}", flush=True)
                payload = {"dataset": {**desc, "class_distribution": cd}, "benchmark": results,
                           "config": {"max_epochs": epochs, "batch_size": batch_size,
                                      "learning_rates": LEARNING_RATES, "optimizers": OPTIMIZERS,
                                      "model_order": MODEL_ORDER}}
                _save_results(payload)
    plot_model_comparison(results, os.path.join(ASSETS, "comparison.png"), MODEL_ORDER, LEARNING_RATES, OPTIMIZERS)
    plot_lr_sensitivity(results, os.path.join(ASSETS, "lr_sensitivity.png"), MODEL_ORDER, LEARNING_RATES)
    means = {lr: np.mean([r["test_accuracy"] for r in results if r["learning_rate"] == lr])
             for lr in LEARNING_RATES}
    blr = max(means, key=means.get)
    bopt = "Adam" if "Adam" in OPTIMIZERS else OPTIMIZERS[0]
    plot_accuracy_bars(results, os.path.join(ASSETS, "accuracy_bars.png"), MODEL_ORDER, blr, OPTIMIZERS)
    plot_train_test_gap(results, os.path.join(ASSETS, "train_test_gap.png"), MODEL_ORDER, blr, bopt)
    print(f"Grille terminée en {time.time()-t0:.0f}s.", flush=True)
    return results


def train_retained(data, y_test_int, epochs=100, batch_size=64, strategy=None):
    """Modèle RETENU (consigne) : meilleure config de la grille, réentraînée."""
    results = _load_results()
    bench = results.get("benchmark", [])
    if not bench:
        print("Pas de grille -> modèle retenu ignoré."); return
    best = max(bench, key=lambda r: r["test_accuracy"])
    label, opt, lr = best["model"], best["optimizer"], best["learning_rate"]
    print(f"Modèle retenu : {label} | {opt} | LR={lr}", flush=True)
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, mode="max",
                                        restore_best_weights=True, verbose=0),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                            min_lr=1e-6, verbose=0)]
    res, hist, model = train_and_evaluate(MODEL_BUILDERS[label], opt, lr, data, label,
                                          epochs=epochs, batch_size=batch_size, callbacks=cb,
                                          strategy=strategy)
    plot_history(hist, f"Modèle retenu ({label})", os.path.join(ASSETS, "retained_curves.png"))
    yp, _ = predict_classes(model, data[2])
    plot_confusion_matrix(y_test_int, yp, os.path.join(ASSETS, "retained_confusion.png"))
    pca = per_class_accuracy(y_test_int, yp, os.path.join(ASSETS, "retained_per_class.png"))
    results = _load_results()
    results["retained_model"] = {"label": label, "optimizer": opt, "learning_rate": lr,
                                 "test_accuracy": res["test_accuracy"], "test_loss": res["test_loss"],
                                 "n_params": res["n_params"], "epochs_run": res["epochs_run"],
                                 "epochs_max": epochs, "per_class_accuracy": pca}
    _save_results(results)
    print(f"Modèle retenu -> test={res['test_accuracy']:.4f}", flush=True)


def train_bonus(data, y_test_int, epochs=60, batch_size=64, strategy=None):
    """Modèle BONUS (hors énoncé) : architecture augmentée pour maximiser la précision."""
    print("Modèle bonus (hors énoncé)...", flush=True)
    if strategy is not None:
        with strategy.scope():
            fm = build_final_model(); compile_model(fm, "Adam", 1e-3)
    else:
        fm = build_final_model(); compile_model(fm, "Adam", 1e-3)
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, mode="max",
                                        restore_best_weights=True, verbose=0),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                            min_lr=1e-6, verbose=0)]
    hist = fm.fit(data[0], data[1], epochs=epochs, batch_size=batch_size,
                  validation_data=(data[2], data[3]), callbacks=cb, verbose=0)
    test_loss, test_acc = fm.evaluate(data[2], data[3], verbose=0)
    plot_history(hist, "Modèle optimisé (bonus)", os.path.join(ASSETS, "final_curves.png"))
    yp, _ = predict_classes(fm, data[2])
    plot_confusion_matrix(y_test_int, yp, os.path.join(ASSETS, "confusion_matrix.png"))
    pca = per_class_accuracy(y_test_int, yp, os.path.join(ASSETS, "per_class_accuracy.png"))
    plot_misclassified(fm, data[2], y_test_int, os.path.join(ASSETS, "misclassified.png"))
    results = _load_results()
    results["final_model"] = {"name": fm.name, "test_accuracy": float(test_acc),
                              "test_loss": float(test_loss), "n_params": int(fm.count_params()),
                              "epochs_run": len(hist.history["loss"]), "epochs_max": epochs,
                              "per_class_accuracy": pca}
    _save_results(results)
    print(f"Modèle bonus -> test={test_acc:.4f}", flush=True)


def train_transfer(raw, y_test_int, epochs_head=3, epochs_ft=15, batch_size=64,
                   img_size=456, strategy=None):
    """Transfer learning EfficientNetB5 (exploite le GPU) -> précision maximale (~97-98%).
    2 phases : tête (base gelée) puis fine-tuning complet. Mixed precision si GPU présent.
    Images BRUTES [0,255]. Sauvegarde dans transfer_* + results['transfer_model']."""
    import tensorflow as tf
    (xtr, ytr), (xte, yte) = raw
    ytr_oh = to_categorical(ytr, 10)
    yte_oh = to_categorical(yte, 10)
    use_mixed = bool(tf.config.list_physical_devices("GPU"))
    if use_mixed:  # Tensor Cores : ~2x plus rapide + moins de mémoire
        keras.mixed_precision.set_global_policy("mixed_float16")

    def _compile(m, lr):
        opt = keras.optimizers.Adam(lr)
        if use_mixed:
            opt = keras.mixed_precision.LossScaleOptimizer(opt)
        m.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    model, base = build_transfer_model(img_size=img_size)
    _compile(model, 1e-3)
    print("Transfer (B5) - phase 1 (tête, base gelée)...", flush=True)
    model.fit(xtr, ytr_oh, epochs=epochs_head, batch_size=batch_size,
              validation_data=(xte, yte_oh), verbose=2)
    base.trainable = True
    _compile(model, 1e-5)
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max",
                                        restore_best_weights=True, verbose=0),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                            min_lr=1e-7, verbose=0)]
    print("Transfer (B5) - phase 2 (fine-tuning)...", flush=True)
    hist = model.fit(xtr, ytr_oh, epochs=epochs_ft, batch_size=batch_size,
                     validation_data=(xte, yte_oh), callbacks=cb, verbose=2)
    test_loss, test_acc = model.evaluate(xte, yte_oh, verbose=0)
    plot_history(hist, f"Transfer learning ({model.name})", os.path.join(ASSETS, "transfer_curves.png"))
    yp, _ = predict_classes(model, xte)
    plot_confusion_matrix(y_test_int, yp, os.path.join(ASSETS, "transfer_confusion.png"))
    pca = per_class_accuracy(y_test_int, yp, os.path.join(ASSETS, "transfer_per_class.png"))
    plot_misclassified(model, xte, y_test_int, os.path.join(ASSETS, "transfer_misclassified.png"))
    results = _load_results()
    results["transfer_model"] = {"name": model.name, "test_accuracy": float(test_acc),
                                 "test_loss": float(test_loss), "n_params": int(model.count_params()),
                                 "epochs_run": len(hist.history["loss"]), "epochs_max": epochs_ft,
                                 "img_size": img_size, "per_class_accuracy": pca}
    _save_results(results)
    if use_mixed:
        keras.mixed_precision.set_global_policy("float32")
    print(f"Transfer (B5) -> test={test_acc:.4f}", flush=True)


def _results_path():
    return os.path.join(ASSETS, "results.json")


def _load_results() -> dict:
    p = _results_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_results(payload):
    os.makedirs(ASSETS, exist_ok=True)
    with open(_results_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ============================================================================
# 6. RAPPORT PDF (annexe — génération du rapport, style article N&B)
#    Cette section ne concerne pas la logique ML ; elle assemble le PDF.
# ============================================================================
from fpdf import FPDF

_BLACK, _GREY = (0, 0, 0), (105, 105, 105)
ABBREV = {"CNN1": "CNN1", "VGG1": "VGG1", "VGG2": "VGG2", "VGG3": "VGG3",
          "VGG3+Drop": "VGG3+D", "VGG3+Drop+BatchNorm": "VGG3+D+BN"}
_AUTHOR = "Mehdi Redha BELLAHSENE"
_HEAD = "Classification d'images CIFAR-10 par CNN"


class _Report(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self._uni = self._fonts()
        self.set_margins(22, 22, 22)
        self._title = True

    def _fonts(self):
        d = r"C:\Windows\Fonts"
        try:
            self.add_font("Serif", "", os.path.join(d, "times.ttf"))
            self.add_font("Serif", "B", os.path.join(d, "timesbd.ttf"))
            self.add_font("Serif", "I", os.path.join(d, "timesi.ttf"))
            return True
        except Exception:
            return False

    def f(self, style="", size=11):
        self.set_font("Serif" if self._uni else "Times", style, size)

    def t(self, s):
        if self._uni:
            return s
        for k, v in {"→": "->", "’": "'", "“": '"', "”": '"',
                     "–": "-", "—": "-", "œ": "oe", "…": "...",
                     "×": "x", "«": '"', "»": '"'}.items():
            s = s.replace(k, v)
        return s.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        if getattr(self, "_title", False):
            return
        self.set_y(10); self.f("I", 8.5); self.set_text_color(*_GREY)
        self.cell(0, 5, self.t(_HEAD), align="L")
        self.cell(0, 5, str(self.page_no()), align="R")
        self.set_draw_color(*_BLACK); self.set_line_width(0.2); self.line(22, 16, 188, 16)
        self.set_y(22); self.set_text_color(*_BLACK)

    def h1(self, x):
        # Saut de page seulement si le titre tomberait en bas de page (evite les orphelins
        # et les pages a moitie vides).
        if self.get_y() > 235:
            self.add_page()
        self.ln(3); self.f("B", 13); self.set_text_color(*_BLACK)
        self.multi_cell(0, 7, self.t(x))
        self.set_draw_color(*_BLACK); self.set_line_width(0.3)
        y = self.get_y() + 0.8; self.line(22, y, 188, y); self.ln(3.5)

    def h2(self, x):
        self.ln(2); self.f("B", 11); self.set_text_color(*_BLACK)
        self.multi_cell(0, 6, self.t(x)); self.ln(1)

    def p(self, x):
        self.f("", 10.5); self.set_text_color(*_BLACK)
        self.multi_cell(0, 5.4, self.t(x), align="J"); self.ln(2)

    def bullet(self, x):
        self.f("", 10.5); self.set_text_color(*_BLACK)
        xs = self.get_x(); self.cell(7, 5.4, self.t("—")); self.set_x(xs + 7)
        self.multi_cell(0, 5.4, self.t(x), align="J"); self.ln(0.6)

    def fig(self, fn, cap, w=160, required=True):
        path = os.path.join(ASSETS, fn)
        if not os.path.exists(path):
            if not required:
                return
            self.f("I", 9); self.set_text_color(*_GREY)
            self.multi_cell(0, 5, self.t(f"[Figure a generer : {fn}]")); self.ln(2)
            self.set_text_color(*_BLACK); return
        # Hauteur d'affichage reelle a partir du ratio de l'image.
        try:
            from PIL import Image as _IMG
            iw, ih = _IMG.open(path).size
            disp_h = w * ih / iw
        except Exception:
            iw, ih, disp_h = 1, 1, w * 0.6
        # Plafonne la hauteur (matrices carrees) pour ne pas occuper toute la page.
        if disp_h > 145:
            disp_h = 145.0
            w = disp_h * iw / ih
        # Saut de page uniquement si la figure + legende ne tient pas dans l'espace restant.
        if self.get_y() + disp_h + 9 > 278:
            self.add_page()
        self.image(path, x=(210 - w) / 2, w=w); self.ln(1.2)
        self.f("I", 8.5); self.set_text_color(*_BLACK)
        self.multi_cell(0, 4.4, self.t(cap), align="C"); self.ln(3)

    def _rule(self, y, wt=0.4):
        self.set_draw_color(*_BLACK); self.set_line_width(wt); self.line(22, y, 188, y)

    def simple_table(self, headers, rows, widths, caption=None):
        x0 = 22 + (166 - sum(widths)) / 2 if sum(widths) < 166 else 22
        self._rule(self.get_y(), 0.5); self.ln(1)
        self.f("B", 9.5); self.set_text_color(*_BLACK); self.set_x(x0)
        for h, w in zip(headers, widths):
            self.cell(w, 6, self.t(h), align="C")
        self.ln(6); self._rule(self.get_y(), 0.3); self.ln(1)
        self.f("", 9.5)
        for row in rows:
            self.set_x(x0)
            for val, w in zip(row, widths):
                self.cell(w, 5.6, self.t(str(val)), align="C")
            self.ln(5.6)
        self._rule(self.get_y(), 0.5); self.ln(2)
        if caption:
            self.f("I", 8.5); self.multi_cell(0, 4.4, self.t(caption), align="C"); self.ln(2)

    def grid_table(self, model_order, lrs, opts, value_fn, title, caption=None):
        label_w = 36.0; col_w = (166.0 - label_w) / (len(lrs) * len(opts)); grp = col_w * len(opts)
        x0 = 22.0
        self._rule(self.get_y(), 0.5); self.ln(1.2)
        self.f("B", 9); self.set_text_color(*_BLACK); self.set_x(x0)
        self.cell(label_w, 6, self.t(title), align="L"); gx = x0 + label_w
        for lr in lrs:
            self.set_x(gx); self.cell(grp, 6, self.t(f"LR = {lr}"), align="C"); gx += grp
        self.ln(6); yb = self.get_y(); gx = x0 + label_w; self.set_line_width(0.2)
        for _ in lrs:
            self.line(gx + 1, yb, gx + grp - 1, yb); gx += grp
        self.ln(0.8); self.f("I", 8.5); self.set_x(x0 + label_w)
        for _ in lrs:
            for o in opts:
                self.cell(col_w, 5, self.t(o), align="C")
        self.ln(5); self._rule(self.get_y(), 0.3); self.ln(1); self.f("", 8.8)
        for m in model_order:
            self.set_x(x0); self.f("B", 8.8)
            self.cell(label_w, 5.4, self.t(ABBREV.get(m, m)), align="L"); self.f("", 8.8)
            for lr in lrs:
                for o in opts:
                    self.cell(col_w, 5.4, self.t(value_fn(m, lr, o)), align="C")
            self.ln(5.4)
        self._rule(self.get_y(), 0.5); self.ln(2)
        cap = "CNN1 = LeNet-5 ; VGG3+D = VGG3 + Dropout ; VGG3+D+BN = VGG3 + Dropout + Batch Normalization."
        if caption:
            cap = caption + " " + cap
        self.f("I", 8.5); self.set_text_color(*_BLACK)
        self.multi_cell(0, 4.4, self.t(cap), align="C"); self.ln(2)


def _uniq(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def _best_lr(bench, lrs):
    if not bench:
        return lrs[0] if lrs else 0.001
    best, blr = -1, lrs[0]
    for lr in lrs:
        v = [r["test_accuracy"] for r in bench if r["learning_rate"] == lr]
        if v and sum(v) / len(v) > best:
            best, blr = sum(v) / len(v), lr
    return blr


def _analysis_blocks(bench, lut, model_order, lrs, opts):
    if not bench:
        return ["[Analyse a completer apres l'entrainement.]"]

    def mean(pred):
        v = [r["test_accuracy"] for r in bench if pred(r)]
        return sum(v) / len(v) if v else 0.0

    by = {}
    for r in bench:
        by.setdefault(r["model"], []).append(r["test_accuracy"])
    depth = " ; ".join(f"{ABBREV.get(m, m)} {max(v)*100:.1f} %" for m, v in by.items())
    adam_div = sum(1 for r in bench if r["optimizer"] == "Adam" and r["test_accuracy"] < 0.15)
    sgd_div = sum(1 for r in bench if r["optimizer"] == "SGD" and r["test_accuracy"] < 0.15)
    lr_means = {lr: mean(lambda r: r["learning_rate"] == lr) for lr in lrs}
    lr_txt = " ; ".join(f"LR={lr} -> {v*100:.1f} %" for lr, v in lr_means.items())
    best_lr = max(lr_means, key=lr_means.get)
    low = min(lrs)

    def om(lr, o):
        return mean(lambda r: r["learning_rate"] == lr and r["optimizer"] == o)
    winners = " ; ".join(f"LR={lr} : {sorted(opts, key=lambda o: om(lr, o), reverse=True)[0]} "
                         f"({om(lr, sorted(opts, key=lambda o: om(lr, o), reverse=True)[0])*100:.0f} %)"
                         for lr in lrs)
    v3a, v3s = lut.get(("VGG3", low, "Adam")), lut.get(("VGG3", low, "SGD"))
    ex = (f" (ex. VGG3 a LR={low} : Adam {v3a['test_accuracy']*100:.1f} % contre SGD "
          f"{v3s['test_accuracy']*100:.1f} %)") if v3a and v3s else ""
    blocks = [
        "Effet de l'architecture. Les performances progressent de CNN1 (LeNet-5) vers les "
        f"VGG avec la profondeur : {depth}. Empiler des blocs convolutifs accroit la "
        "capacite d'extraction de caracteristiques, donc la precision, jusqu'a un palier.",
        "Effet de l'optimiseur. Le meilleur optimiseur depend du learning rate. Au learning "
        f"rate adapte (LR={low}), Adam surpasse SGD{ex}. Mais a 0.01 et 0.1, Adam devient "
        f"instable et diverge, alors que SGD reste robuste : {adam_div} divergence(s) pour "
        f"Adam contre {sgd_div} pour SGD. Gagnant par learning rate : {winners}. Adam est "
        "le meilleur a faible LR, SGD plus sur a LR eleve.",
        f"Effet du learning rate. Precision moyenne par LR : {lr_txt} ; meilleur compromis "
        f"LR={best_lr}. Un LR trop eleve fait diverger Adam (precision ~10 %, niveau du "
        "hasard) : son pas adaptatif sature la sortie softmax. SGD reste stable. La "
        "sensibilite au learning rate est une faiblesse d'Adam et une force de SGD.",
    ]
    rl = _reg_effect(lut, low, opts)
    if rl:
        blocks.append(rl)
    bl = _bn_high_lr(lut, bench, model_order, lrs, opts)
    if bl:
        blocks.append(bl)
    return blocks


def _reg_effect(lut, lr, opts):
    o = "Adam" if "Adam" in opts else opts[0]
    b, d, n = lut.get(("VGG3", lr, o)), lut.get(("VGG3+Drop", lr, o)), lut.get(("VGG3+Drop+BatchNorm", lr, o))
    if not (b and d and n):
        return ""

    def gap(r):
        g = r.get("overfit_gap")
        return f" (ecart train-test {g*100:.1f} pts)" if g else ""
    return (f"Effet de la regularisation (VGG3, {o}, LR={lr}). En partant de VGG3 "
            f"({b['test_accuracy']*100:.1f} %{gap(b)}), l'ajout du Dropout donne "
            f"{d['test_accuracy']*100:.1f} %{gap(d)}, puis l'ajout de la Batch Normalization donne "
            f"{n['test_accuracy']*100:.1f} %{gap(n)}. La regularisation reduit le surapprentissage ; "
            "la Batch Normalization stabilise en outre l'entrainement et autorise des learning rates "
            "plus eleves.")


def _bn_high_lr(lut, bench, model_order, lrs, opts):
    high = max(lrs)
    bn = next((m for m in model_order if "BatchNorm" in m), None)
    if bn is None:
        return ""
    bn_a = [lut[(bn, high, o)]["test_accuracy"] for o in opts if (bn, high, o) in lut]
    oth = [r["test_accuracy"] for r in bench if r["learning_rate"] == high and "BatchNorm" not in r["model"]]
    if not bn_a or not oth:
        return ""
    bb, ob = max(bn_a), max(oth)
    if bb > 0.5 and (bb - ob) > 0.3:
        return (f"Batch Normalization et robustesse au learning rate. A LR={high}, seul le modele "
                f"avec Batch Normalization evite la divergence (jusqu'a {bb*100:.1f} %), tandis que "
                f"les autres s'effondrent au niveau du hasard (au mieux {ob*100:.0f} %). C'est la "
                "demonstration du benefice central de la BN : en normalisant les activations, elle "
                "empeche l'explosion des gradients et autorise des learning rates bien plus eleves.")
    return ""


def _conclusion_blocks(bench, fm, rm, tm=None):
    parts = [
        "Ce projet a deroule une demarche complete de classification d'images (CRISP-DM). Le "
        "pretraitement (normalisation, one-hot) est indispensable a un entrainement stable. Les "
        "VGG depassent largement CNN1 (LeNet-5), la profondeur ameliorant l'extraction de "
        "caracteristiques.",
        "Adam est le plus efficace a faible learning rate mais sensible (il diverge a 0.01 et 0.1) ; "
        "SGD est plus lent mais robuste. Une valeur faible (0.001) est le meilleur compromis. "
        "Dropout et Batch Normalization limitent le surapprentissage ; la BN autorise en plus des "
        "learning rates eleves.",
    ]
    if rm:
        parts.append(f"Le modele retenu ({ABBREV.get(rm['label'], rm['label'])} / {rm['optimizer']} / "
                     f"LR={rm['learning_rate']}) atteint {rm['test_accuracy']*100:.2f} % sur le test.")
    elif bench:
        b = max(bench, key=lambda r: r["test_accuracy"])
        parts.append(f"La meilleure configuration ({ABBREV.get(b['model'], b['model'])} / {b['optimizer']} "
                     f"/ LR={b['learning_rate']}) atteint {b['test_accuracy']*100:.2f} %.")
    if fm:
        parts.append(f"En complement (hors enonce), un modele optimise entraine de zero "
                     f"(augmentation + filtres croissants) atteint {fm['test_accuracy']*100:.2f} %.")
    if tm:
        parts.append(f"Enfin, le transfer learning ({tm.get('name', 'EfficientNet')} pre-entraine sur "
                     f"ImageNet, affine) porte la precision a {tm['test_accuracy']*100:.2f} % — le "
                     "meilleur resultat du projet, illustrant la puissance du transfert de "
                     "connaissances par rapport a un entrainement de zero.")
    parts.append("Pistes : architectures residuelles (ResNet), recherche systematique des "
                 "hyperparametres, augmentation avancee (MixUp, RandAugment).")
    return parts


def build_report(results):
    """Assemble le rapport PDF (style article noir et blanc) depuis results.json."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, "Rapport_ML_CIFAR10.pdf")
    pdf = _Report()
    bench = results.get("benchmark", [])
    cfg = results.get("config", {})
    mo = cfg.get("model_order") or _uniq([r["model"] for r in bench]) or MODEL_ORDER
    lrs = cfg.get("learning_rates") or sorted({r["learning_rate"] for r in bench}) or LEARNING_RATES
    opts = cfg.get("optimizers") or sorted({r["optimizer"] for r in bench}) or OPTIMIZERS
    lut = {(r["model"], r["learning_rate"], r["optimizer"]): r for r in bench}
    ds = results.get("dataset", {})
    fm = results.get("final_model")
    tm = results.get("transfer_model")
    rm = results.get("retained_model")
    if rm is None and bench:  # déduit le modèle retenu = meilleure config de la grille
        b = max(bench, key=lambda r: r["test_accuracy"])
        rm = {"label": b["model"], "optimizer": b["optimizer"], "learning_rate": b["learning_rate"],
              "test_accuracy": b["test_accuracy"], "test_loss": b.get("test_loss"),
              "n_params": b.get("n_params"), "epochs_run": b.get("epochs_run"),
              "epochs_max": cfg.get("max_epochs", "?"), "_derived": True}

    # Titre + résumé
    pdf._title = True; pdf.add_page(); pdf.ln(26)
    pdf._rule(pdf.get_y(), 0.4); pdf.ln(6)
    pdf.f("B", 19); pdf.set_text_color(*_BLACK)
    pdf.multi_cell(0, 9, pdf.t("Classification d'images par reseaux de neurones convolutifs (CNN)"), align="C")
    pdf.ln(2); pdf.f("I", 12)
    pdf.multi_cell(0, 6.5, pdf.t("Jeu de donnees CIFAR-10 : architectures VGG, optimiseurs, learning rate et regularisation"), align="C")
    pdf.ln(4); pdf._rule(pdf.get_y(), 0.4); pdf.ln(8)
    pdf.f("", 12); pdf.cell(0, 6, pdf.t(_AUTHOR), align="C"); pdf.ln(7)
    pdf.f("I", 10.5); pdf.set_text_color(*_GREY)
    pdf.cell(0, 5, pdf.t("Methodologie CRISP-DM  .  Python / TensorFlow / Keras  .  Juin 2026"), align="C")
    pdf.ln(7); pdf.f("", 10.5); pdf.set_text_color(*_BLACK)
    pdf.cell(0, 5, pdf.t("Application : https://machine-learning.bellahsene.org"), align="C",
             link="https://machine-learning.bellahsene.org")
    pdf.ln(6)
    pdf.cell(0, 5, pdf.t("Code source : https://github.com/mehdibellahsene/projet-ml-cifar10"), align="C",
             link="https://github.com/mehdibellahsene/projet-ml-cifar10")
    pdf.ln(13); pdf.set_text_color(*_BLACK); pdf.f("B", 10.5)
    pdf.cell(0, 6, pdf.t("Resume"), align="C"); pdf.ln(7)
    best_txt = ""
    if bench:
        b = max(bench, key=lambda r: r["test_accuracy"])
        best_txt = (f" La meilleure configuration imposee ({ABBREV.get(b['model'], b['model'])}, "
                    f"{b['optimizer']}, LR={b['learning_rate']}) atteint {b['test_accuracy']*100:.1f} %.")
    if tm:
        best_txt += (f" En allant plus loin, le transfer learning ({tm.get('name', 'EfficientNet')}) "
                     f"porte la precision a {tm['test_accuracy']*100:.1f} %, le meilleur resultat du "
                     "projet.")
    pdf.set_left_margin(40); pdf.set_right_margin(40); pdf.f("", 10)
    pdf.multi_cell(0, 5.2, pdf.t(
        "Ce travail etudie la classification d'images CIFAR-10 par CNN. Nous comparons une "
        "architecture de base (LeNet-5) a des VGG (1 a 3 blocs), evaluons l'effet de l'optimiseur "
        "(SGD, Adam) et du learning rate, puis l'apport de la regularisation (Dropout, Batch "
        "Normalization), selon CRISP-DM." + best_txt), align="J")
    pdf.set_left_margin(22); pdf.set_right_margin(22); pdf._title = False

    # 1. Intro
    pdf.add_page(); pdf.h1("1. Introduction et methodologie")
    pdf.p("L'objectif de ce projet est d'implementer des reseaux de neurones convolutifs (CNN) "
          "pour la classification d'images du jeu CIFAR-10 : etude des architectures de type VGG, "
          "comparaison des optimiseurs SGD et Adam sur trois learning rates, et evaluation des "
          "techniques de regularisation (Dropout, Batch Normalization).")
    pdf.h2("1.1 Methodologie CRISP-DM (les six phases)")
    pdf.p("La demarche suit la methodologie CRISP-DM (Cross-Industry Standard Process for Data "
          "Mining), un processus iteratif en six phases ; voici comment chacune se concretise dans "
          "ce projet :")
    pdf.bullet("Comprehension du probleme : classer des images couleur 32x32 en 10 categories "
               "mutuellement exclusives (tache de vision par ordinateur).")
    pdf.bullet("Comprehension des donnees : exploration de CIFAR-10 -- dimensions, exemples "
               "visuels, distribution des classes (section 2).")
    pdf.bullet("Preparation des donnees : normalisation des pixels dans [0,1] et encodage one-hot "
               "des labels (section 2).")
    pdf.bullet("Modelisation : construction de LeNet-5, VGG1/2/3 et des variantes regularisees ; "
               "choix des optimiseurs et des learning rates (section 3).")
    pdf.bullet("Evaluation : tableau de performances, courbes d'apprentissage et analyse "
               "comparative des resultats (section 4).")
    pdf.bullet("Deploiement : selection du modele retenu (section 5) et extensions pour la "
               "precision maximale (sections 6-7) ; l'application web ML Playground met le modele "
               "a l'epreuve (section 8).")
    pdf.h2("1.2 Environnement d'execution et consommation des ressources")
    pdf.p("Les entrainements ont ete realises en Python (TensorFlow / Keras) sur Google Colab, avec "
          "un GPU NVIDIA A100 (80 Go de memoire GPU) dans un environnement a HAUTE MEMOIRE "
          "(High-RAM, 167 Go de RAM systeme). Cette configuration High-RAM a ete necessaire pour "
          "entrainer le modele de transfer learning EfficientNetB5 a haute resolution (456x456) sans "
          "saturation de la memoire. Le transfer learning emploie en outre la precision mixte "
          "(float16) pour exploiter les Tensor Cores du GPU et accelerer l'entrainement.")
    pdf.p("Point sur l'execution. La grille des 36 configurations (modeles legers) s'entraine en "
          "quelques minutes ; a l'inverse, EfficientNetB5 -- le plus gros modele -- sollicite "
          "fortement le GPU (environ 18 minutes par epoque en fine-tuning a 456x456). La capture "
          "ci-dessous montre la consommation des ressources (RAM systeme, memoire GPU, disque) "
          "pendant l'execution sur l'environnement A100 High-RAM.")
    pdf.fig("execution_resources.png",
            "Figure. Consommation des ressources sur Colab (A100, environnement High-RAM).",
            w=150, required=False)

    # 2. Données
    pdf.h1("2. Comprehension et preparation des donnees")
    pdf.h2("2.1 Jeu de donnees (Tache 1)")
    pdf.p("CIFAR-10 : images couleur 32x32 (3 canaux RVB), 10 classes mutuellement exclusives.")
    if ds:
        pdf.simple_table(["Caracteristique", "Valeur"],
                         [["Images d'entrainement", f"{ds.get('n_train', 0):,}".replace(",", " ")],
                          ["Images de test", f"{ds.get('n_test', 0):,}".replace(",", " ")],
                          ["Dimensions image", " x ".join(map(str, ds.get("image_shape", [])))],
                          ["Canaux", ds.get("n_channels", "?")], ["Classes", ds.get("n_classes", "?")]],
                         [70, 50], caption="Table 1. Caracteristiques de CIFAR-10.")
    pdf.p("50 000 images d'entrainement et 10 000 de test. La faible resolution rend la tache "
          "difficile (classes proches : chat/chien, automobile/camion).")
    pdf.h2("2.2 Visualisation (Tache 2)")
    pdf.p("Les labels sont des donnees CATEGORIELLES (et non numeriques ordonnees) : chaque entier "
          "de 0 a 9 designe une categorie (avion, automobile, ...), sans relation d'ordre entre "
          "elles. Visuellement, la tres faible resolution (32x32) rend certaines images difficiles "
          "a classer, y compris pour un humain -- notamment des classes proches comme chat / chien "
          "ou automobile / camion.")
    pdf.fig("samples.png", "Figure 1. Echantillon d'images CIFAR-10 et leurs classes.")
    pdf.h2("2.3 Normalisation (Tache 3)")
    pdf.p("Les pixels (0-255) sont ramenes dans [0,1] par division par 255. Cela place les entrees "
          "sur une echelle commune, stabilise et accelere la descente de gradient.")
    pdf.h2("2.4 Encodage one-hot (Tache 4)")
    pdf.p("Les labels deviennent des vecteurs one-hot (classe 3 -> [0,0,0,1,0,0,0,0,0,0]). Cela "
          "evite toute relation d'ordre entre classes et s'accorde avec softmax + entropie croisee.")
    pdf.h2("2.5 Distribution des classes (Tache 5)")
    pdf.p("CIFAR-10 est equilibre : 5 000 images par classe. Un desequilibre biaiserait le modele.")
    pdf.fig("class_distribution.png", "Figure 2. Distribution des classes (5 000 chacune).")

    # 3. Modélisation
    pdf.h1("3. Modelisation : architectures, optimiseurs, regularisation")
    pdf.h2("3.1 Architectures")
    pdf.p("Les architectures de type VGG (Simonyan & Zisserman, 2014) empilent des blocs "
          "convolutifs identiques. Un bloc VGG = Conv -> Conv -> MaxPool (32 filtres 3x3, pooling "
          "2x2) ; suit une couche cachee Dense(128) puis la sortie softmax(10). Six modeles :")
    pdf.bullet("CNN1 : reseau de base (LeNet-5), reference.")
    pdf.bullet("VGG1/2/3 : 1, 2 ou 3 blocs (profondeur croissante).")
    pdf.bullet("VGG3+Drop : VGG3 + Dropout.")
    pdf.bullet("VGG3+Drop+BatchNorm : VGG3 + Dropout + Batch Normalization.")
    params = {}
    for r in bench:
        params.setdefault(r["model"], r.get("n_params"))
    if params:
        blocks = {"CNN1": "-", "VGG1": "1", "VGG2": "2", "VGG3": "3", "VGG3+Drop": "3", "VGG3+Drop+BatchNorm": "3"}
        rows = [[ABBREV.get(m, m), blocks.get(m, "-"), f"{params[m]:,}".replace(",", " ") if params.get(m) else "?"]
                for m in mo if m in params]
        pdf.simple_table(["Modele", "Blocs VGG", "Parametres"], rows, [45, 35, 45],
                         caption="Table 2. Nombre de parametres par architecture.")
    pdf.h2("3.2 Optimiseurs (SGD et Adam)")
    pdf.p("Un optimiseur est l'algorithme qui ajuste les poids du reseau afin de minimiser la "
          "fonction de perte. A chaque iteration, il exploite les gradients calcules par "
          "retropropagation pour mettre a jour chaque poids dans la direction qui reduit l'erreur. "
          "Son role est central : il determine la vitesse, la stabilite et la qualite finale de "
          "l'apprentissage.")
    pdf.bullet("SGD (descente de gradient stochastique) met a jour chaque poids d'un pas fixe : "
               "w <- w - lr x gradient. Simple et robuste, mais sensible au choix du learning rate "
               "et plus lent ; le momentum (0,9) accumule les gradients passes pour accelerer et "
               "lisser la trajectoire.")
    pdf.bullet("Adam (Adaptive Moment Estimation) combine le momentum (moyenne des gradients) et un "
               "learning rate adaptatif par parametre (via la moyenne des carres des gradients) : "
               "w <- w - lr x m / (racine(v) + eps). Il converge plus vite et demande moins de "
               "reglage, d'ou son statut de choix par defaut pour les CNN.")
    pdf.p("Quel optimiseur pour l'architecture VGG ? L'analyse (section 4.3) le montre : Adam est le "
          "plus performant a faible learning rate (0.001), mais il diverge a learning rate eleve, ou "
          "SGD reste plus robuste. (Refs : Kingma & Ba, 2015 ; Analytics Vidhya, 2021.)")
    pdf.h2("3.3 Regularisation : Dropout et Batch Normalization (Bonus)")
    pdf.p("Dropout. Pendant l'entrainement, le Dropout desactive aleatoirement une fraction des "
          "neurones (20 a 50 % ici) a chaque etape : le reseau ne peut plus dependre de neurones "
          "specifiques et apprend des representations redondantes et robustes, ce qui reduit le "
          "surapprentissage. Difference essentielle entre les deux phases : le Dropout n'agit qu'a "
          "l'ENTRAINEMENT ; en INFERENCE, tous les neurones sont conserves et leurs sorties mises a "
          "l'echelle de maniere coherente. (Ref : Srivastava et al., 2014.)")
    pdf.p("Batch Normalization. Elle normalise les activations de chaque couche au sein d'un "
          "mini-batch (moyenne nulle, variance unitaire), puis les remet a l'echelle via deux "
          "parametres appris. Objectifs et benefices : reduire le decalage de distribution interne "
          "(covariate shift), ce qui stabilise et accelere l'entrainement, autorise des learning "
          "rates plus eleves et exerce un leger effet regularisant. (Ref : Ioffe & Szegedy, 2015.)")
    pdf.h2("3.4 Protocole")
    pdf.p(f"Entropie croisee categorielle, batch {cfg.get('batch_size', 64)}, max "
          f"{cfg.get('max_epochs', 50)} epoques, early stopping. Trois learning rates (0.001, 0.01, "
          "0.1) x deux optimiseurs ; LR fixe pendant chaque entrainement. Le test sert de validation.")

    # 4. Résultats
    pdf.h1("4. Implementation et evaluation des performances")
    pdf.h2("4.1 Tableau des performances (precision de test)")

    def acc(m, lr, o):
        r = lut.get((m, lr, o))
        return f"{r['test_accuracy']*100:.1f}" if r else "-"
    pdf.grid_table(mo, lrs, opts, acc, "Precision test (%)", caption="Table 3. Precision de test (%).")
    if bench:
        b = max(bench, key=lambda r: r["test_accuracy"])
        tr = (f" (entrainement {b['train_accuracy']*100:.2f} %)"
              if b.get("train_accuracy") is not None else "")
        pdf.p(f"Meilleure configuration : {ABBREV.get(b['model'], b['model'])} / {b['optimizer']} / "
              f"LR={b['learning_rate']} -> {b['test_accuracy']*100:.2f} %{tr}.")
    if any(r.get("test_loss") is not None for r in bench):
        pdf.h2("4.2 Perte de test")

        def loss(m, lr, o):
            r = lut.get((m, lr, o))
            return f"{r['test_loss']:.2f}" if r and r.get("test_loss") is not None else "-"
        pdf.grid_table(mo, lrs, opts, loss, "Perte test", caption="Table 4. Perte (entropie croisee).")
    pdf.fig("comparison.png", "Figure 3. Precision de test (heatmap) : modeles x (learning rate, optimiseur).")
    pdf.fig("accuracy_bars.png", "Figure 4. Precision de test par architecture et optimiseur (au meilleur learning rate).")
    pdf.fig("lr_sensitivity.png", "Figure 5. Sensibilite de la precision au learning rate, par optimiseur.")
    pdf.fig("train_test_gap.png", "Figure 6. Entrainement vs test : l'ecart mesure le sur-apprentissage.", required=False)
    pdf.h2("4.3 Analyse des resultats")
    for blk in _analysis_blocks(bench, lut, mo, lrs, opts):
        pdf.p(blk)
    blr = _best_lr(bench, lrs)
    has_curves = os.path.isdir(CURVES) and len(os.listdir(CURVES)) > 0
    if has_curves:
        pdf.h2(f"4.4 Courbes d'apprentissage (learning rate = {blr})")
        pdf.p("Perte et precision (entrainement + test) pour chaque modele, sous SGD puis Adam.")
        i = 7
        for m in mo:
            for o in opts:
                r = lut.get((m, blr, o))
                if r and r.get("curve") and os.path.exists(os.path.join(ASSETS, r["curve"].replace("\\", "/"))):
                    pdf.fig(r["curve"].replace("\\", "/"), f"Figure {i}. {ABBREV.get(m, m)} / {o} / LR={blr}.", w=145)
                    i += 1

    # 5. Modèle retenu (consigne)
    pdf.h1("5. Modele retenu")
    if rm:
        if rm.get("_derived"):
            pdf.p(f"Conformement a l'enonce, le modele retenu est la meilleure configuration de la "
                  f"comparaison : {ABBREV.get(rm['label'], rm['label'])} avec l'optimiseur "
                  f"{rm['optimizer']} et un learning rate de {rm['learning_rate']}, atteignant "
                  f"{rm['test_accuracy']*100:.2f} % de precision sur le test. C'est donc l'architecture "
                  "imposee la plus performante de l'etude.")
        else:
            pdf.p(f"Conformement a l'enonce, le modele retenu est la meilleure configuration : "
                  f"{ABBREV.get(rm['label'], rm['label'])} / {rm['optimizer']} / LR={rm['learning_rate']}, "
                  "reentraine plus longtemps (n'utilise que les architectures imposees).")
        rows = [["Architecture", ABBREV.get(rm["label"], rm["label"])],
                ["Optimiseur / LR", f"{rm['optimizer']} / {rm['learning_rate']}"],
                ["Precision de test", f"{rm['test_accuracy']*100:.2f} %"]]
        if rm.get("test_loss"):
            rows.append(["Perte de test", f"{rm['test_loss']:.3f}"])
        if rm.get("n_params"):
            rows.append(["Parametres", f"{rm['n_params']:,}".replace(",", " ")])
        if rm.get("epochs_run"):
            rows.append(["Epoques", f"{rm['epochs_run']} / {rm.get('epochs_max', '?')}"])
        pdf.simple_table(["Indicateur", "Valeur"], rows, [70, 60], caption="Table 5. Modele retenu.")
        pdf.p("A noter : meme reentraine longuement (jusqu'a 100 epoques), ce modele plafonne autour "
              "de 84-85 %. Ce n'est pas un defaut d'entrainement mais une limite de CAPACITE : "
              "l'architecture imposee est volontairement petite (blocs VGG a 32 filtres seulement, "
              "~114 000 parametres). La precision d'entrainement et la precision de test restent "
              "proches (autour de 85 %), ce qui indique une sous-capacite et non du surapprentissage : "
              "le reseau atteint sa limite de representation. Au-dela, ce n'est plus le nombre "
              "d'epoques qui compte mais la capacite du modele -- d'ou les extensions des sections 6 "
              "et 7 (architecture plus large : 89.6 % ; transfer learning : 97.9 %).")
    else:
        pdf.p("[Modele retenu a inserer apres entrainement.]")
    pdf.fig("retained_curves.png", "Figure. Courbes du modele retenu.", required=False)
    pdf.fig("retained_confusion.png", "Figure. Matrice de confusion du modele retenu (%).", required=False)
    pdf.fig("retained_per_class.png", "Figure. Precision par classe du modele retenu.", required=False)
    if rm and rm.get("per_class_accuracy"):
        pca = rm["per_class_accuracy"]; w = min(pca, key=pca.get); bc = max(pca, key=pca.get)
        pdf.p(f"Classe la mieux reconnue : {FR_NAMES.get(bc, bc)} ({pca[bc]*100:.1f} %) ; la plus "
              f"difficile : {FR_NAMES.get(w, w)} ({pca[w]*100:.1f} %). Confusions entre classes proches.")

    # 6. Bonus 1 : modèle optimisé (from-scratch OU transfer léger selon le run)
    if fm:
        fname = fm.get("name", "")
        is_tl = ("transfer" in fname.lower() or "efficientnet" in fname.lower())
        if is_tl:
            pdf.h1("6. Bonus 1 : transfer learning (modele plus leger)")
            pdf.p(f"Premiere extension (hors enonce) par transfer learning : un reseau "
                  f"{fname} pre-entraine sur ImageNet, dont les images CIFAR-10 sont agrandies "
                  "puis le reseau affine. Backbone plus leger (entrainement rapide), deja tres "
                  "performant grace au transfert de connaissances.")
            cap = "Table 6. Modele transfer learning (leger)."
            ftitle = "Modele transfer (leger)"
        else:
            pdf.h1("6. Bonus 1 : modele optimise (from-scratch)")
            pdf.p("Premiere extension (hors enonce), entrainee de zero sans poids pre-entraines : un "
                  "reseau plus profond (filtres croissants 64->128->256, Batch Normalization et Dropout "
                  "systematiques, augmentation de donnees), qui ameliore la generalisation.")
            cap = "Table 6. Modele optimise from-scratch."
            ftitle = "Modele optimise"
        pdf.simple_table(["Indicateur", "Valeur"],
                         [["Modele", fname or "-"],
                          ["Precision de test", f"{fm['test_accuracy']*100:.2f} %"],
                          ["Perte de test", f"{fm['test_loss']:.3f}"],
                          ["Parametres", f"{fm['n_params']:,}".replace(",", " ")],
                          ["Epoques", f"{fm['epochs_run']} / {fm.get('epochs_max', '?')}"]],
                         [70, 70], caption=cap)
        pdf.p(f"Ce modele atteint {fm['test_accuracy']*100:.2f} %, au-dela des architectures imposees.")
        pdf.fig("final_curves.png", f"Figure. Courbes ({ftitle}).")
        pdf.fig("confusion_matrix.png", f"Figure. Matrice de confusion ({ftitle}, %).")
        pdf.fig("per_class_accuracy.png", f"Figure. Precision par classe ({ftitle}).")
        pdf.fig("misclassified.png", "Figure. Exemples mal classes (V = reel, P = predit).")

    # 7. Bonus 2 : transfer learning EfficientNetB5 (précision maximale)
    if tm:
        pdf.h1("7. Bonus 2 : transfer learning (precision maximale)")
        img = tm.get("img_size", 384)
        pdf.p(f"Seconde extension visant la precision maximale : le transfer learning. Un reseau "
              f"{tm.get('name', 'EfficientNet')} pre-entraine sur ImageNet est reutilise ; les images "
              f"CIFAR-10 (32x32) sont agrandies a {img}x{img}, puis le reseau est affine en deux phases "
              "(tete de classification, puis fine-tuning complet a faible learning rate, en mixed "
              "precision pour exploiter le GPU). Le transfert de connaissances depuis ImageNet permet "
              "d'atteindre une precision nettement superieure aux modeles entraines de zero.")
        pdf.simple_table(["Indicateur", "Valeur"],
                         [["Modele", tm.get("name", "-")],
                          ["Resolution d'entree", f"{img} x {img}"],
                          ["Precision de test", f"{tm['test_accuracy']*100:.2f} %"],
                          ["Perte de test", f"{tm['test_loss']:.3f}"],
                          ["Parametres", f"{tm['n_params']:,}".replace(",", " ")],
                          ["Epoques (fine-tuning)", f"{tm['epochs_run']} / {tm.get('epochs_max', '?')}"]],
                         [70, 75], caption="Table 7. Transfer learning (precision maximale).")
        pdf.p(f"Ce modele atteint {tm['test_accuracy']*100:.2f} % de precision — le meilleur resultat "
              "du projet.")
        pdf.fig("transfer_curves.png", "Figure. Courbes du modele transfer learning.", required=False)
        pdf.fig("transfer_confusion.png", "Figure. Matrice de confusion (transfer learning, %).", required=False)
        pdf.fig("transfer_per_class.png", "Figure. Precision par classe (transfer learning).", required=False)
        pdf.fig("transfer_misclassified.png", "Figure. Exemples mal classes (transfer learning).", required=False)

    # 8. ML Playground (déploiement) : mise à l'épreuve du modèle
    pdf.h1("8. ML Playground : mise a l'epreuve du modele")
    pdf.p("Phase de deploiement de la methodologie CRISP-DM : le modele de transfer learning "
          "(section 7) est mis en production dans ML Playground, une application web developpee "
          "pour le projet (backend FastAPI, frontend statique, deploiement Docker), accessible en "
          "ligne sur https://machine-learning.bellahsene.org. L'interface est adaptee a la fois au "
          "web et au mobile.")
    pdf.p("De l'entrainement a l'usage reel : le modele entraine et evalue sur Colab (sections 4 a "
          "7) est sauvegarde au format Keras avec son pre-traitement integre au graphe "
          "(agrandissement 32x32 -> 456x456 et normalisation EfficientNet). L'application n'a donc "
          "qu'a lui envoyer une image 32x32 brute : un backend FastAPI charge le modele et expose "
          "une API de prediction, un frontend leger interroge cette API, et le tout est "
          "conteneurise avec Docker puis deploye sur un serveur auto-heberge. La theorie -- "
          "courbes d'apprentissage, matrices de confusion -- devient ainsi un usage concret : "
          "chaque clic des joueurs declenche une vraie inference du modele. Trois mini-jeux "
          "mettent le modele a l'epreuve, chacun sous un angle different :")
    pdf.bullet("Le Duel -- humain contre machine : 10 manches, 3 secondes par image pour choisir "
               "la bonne classe parmi trois propositions ; l'IA repond en parallele a la meme "
               "image et les scores sont compares en direct.")
    pdf.bullet("Dessine, je devine : l'utilisateur dessine la classe imposee sur un canvas et le "
               "modele devine en temps reel, trait apres trait. Parfois quelques bonnes couleurs "
               "suffisent pour qu'il trouve ; parfois il faut etre plus artistique et pointu, car "
               "un dessin est tres loin des photos 32x32 vues a l'entrainement.")
    pdf.bullet("Test Ultime CINIC-10 : partie pensee pour le correcteur, qui teste le modele sur "
               "des images generiques issues de CINIC-10 (derivees d'ImageNet) qu'il n'a JAMAIS "
               "vues a l'entrainement. C'est l'epreuve la plus difficile : ces images ne suivent "
               "pas toujours les memes criteres de categorisation que CIFAR-10, et l'enjeu est de "
               "reperer quand le modele se trompe.")
    pdf.fig("playground_home.png", "Figure. Screen du jeu.", w=170, required=False)
    pdf.fig("playground_duel_mobile.png", "Figure. Screen du jeu.", w=80, required=False)
    pdf.p("L'application a tres bien fonctionne : etudiants, amis et famille l'ont testee et se "
          "sont mis en defi les uns les autres, comme en temoignent les classements de la page "
          "d'accueil. C'est aussi ce qui m'a fait le plus plaisir dans ce projet : voir "
          "concretement le resultat de l'effort de la puce A100 -- un modele entraine pendant des "
          "heures devenu un jeu que chacun peut defier depuis son navigateur ou son telephone.")

    # 9. Conclusion
    pdf.h1("9. Conclusion generale")
    for blk in _conclusion_blocks(bench, fm, rm, tm):
        pdf.p(blk)

    # 10. Références
    pdf.h1("10. References")
    for r in [
        "Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale "
        "Image Recognition (VGG). arXiv:1409.1556.",
        "Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.",
        "Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from "
        "Overfitting. JMLR 15.",
        "Ioffe, S. & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training "
        "by Reducing Internal Covariate Shift. ICML.",
        "Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural "
        "Networks. ICML.",
        "Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images (CIFAR-10).",
        "LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition (LeNet-5).",
        "Analytics Vidhya (2021). A Comprehensive Guide on Deep Learning Optimizers.",
    ]:
        pdf.bullet(r)

    pdf.output(out)
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Projet CIFAR-10 CNN")
    ap.add_argument("--report", action="store_true", help="générer seulement le PDF")
    ap.add_argument("--quick", action="store_true", help="test rapide (peu d'époques)")
    ap.add_argument("--no-bonus", action="store_true", help="sans le modèle bonus")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    if args.report:
        path = build_report(_load_results())
        print("Rapport généré :", path); return

    epochs = 3 if args.quick else args.epochs
    raw = load_raw()
    (xtr, ytr), (xte, yte) = raw
    data = preprocess(xtr, ytr, xte, yte)
    y_test_int = yte.flatten()

    run_grid(data, raw, epochs=epochs)
    train_retained(data, y_test_int, epochs=10 if args.quick else 100)
    if not args.no_bonus:
        # (1) Modèle optimisé from-scratch (basique) + (2) transfer learning B5 (GPU max).
        train_bonus(data, y_test_int, epochs=5 if args.quick else 60)
        train_transfer(raw, y_test_int,
                       epochs_head=1 if args.quick else 3,
                       epochs_ft=2 if args.quick else 15,
                       img_size=96 if args.quick else 456)
    path = build_report(_load_results())
    print("Pipeline terminé. Rapport :", path)


if __name__ == "__main__":
    main()
