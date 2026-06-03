# ML Model Detection — CIFAR-10

Projet complet de classification d'images **CIFAR-10** avec des réseaux de
neurones convolutifs, suivant la méthodologie **CRISP-DM** : de l'analyse des
données jusqu'à une application web jouable.

**Application en ligne : https://machine-learning.bellahsene.org**

## Contenu du dépôt

| Élément | Description |
|---------|-------------|
| `TP_Project_ML.ipynb` | Notebook complet (Colab) : données, modèles, entraînement, résultats. |
| `projet_cifar10.py` | Script Python autonome (même logique que le notebook) + génération du rapport. |
| `report/Rapport_ML_CIFAR10.pdf` | Rapport en français (logique + résultats, style article). |
| `report_assets/` | Figures et résultats (`results.json`) produits par l'entraînement. |
| `playground-cifar-model/` | **Application web** (FastAPI + 3 mini-jeux) qui expose le modèle. |

## Résultats clés

- Grille comparative LeNet-5 / VGG1 / VGG2 / VGG3 × {SGD, Adam} × régularisation.
- Meilleure config « from scratch » retenue : VGG3 + Dropout + BatchNorm (**~85 %**).
- **Transfer learning EfficientNetB5 : ~97,9 %** (modèle servi par l'application).

## L'application

Trois mini-jeux qui tapent sur le modèle entraîné :

1. **Le Duel : Humain vs Machine** — 3 s par image, score comparatif sur 10 manches.
2. **Crash Test Pictionary** — dessine, l'IA devine en temps réel.
3. **Test Ultime CINIC-10** — images jamais vues à l'entraînement.

Détails techniques et déploiement : voir [`playground-cifar-model/README.md`](playground-cifar-model/README.md).

## Auteur

Mehdi Redha BELLAHSENE
