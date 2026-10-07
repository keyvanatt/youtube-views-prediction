# Y-VAMPX — YouTube Views Attention-based Multimodal Predictor

Projet du Data Challenge **CSC_43M04_EP** (Modal d'informatique, École polytechnique) — Louis Darrigol & Keyvan Attarian.

L'objectif est de prédire le nombre de vues d'une vidéo YouTube à partir de sa **miniature**, de son **titre**, de sa **description** et de métadonnées (chaîne, année). Le modèle prédit `log(1 + views)`.

## Données

- `dataset/train_val.csv` : ~15k vidéos (2011–2023), `dataset/test.csv` : ~3k vidéos (2024 – début 2025).
- Les miniatures (non versionnées) sont attendues dans `dataset/train_val/<id>.jpg` et `dataset/test/<id>.jpg`.
- Colonnes ajoutées au dataset d'origine : `channel_id` (chaîne encodée en entier, 46 chaînes), `log1p_views`, `category` (pour le classifieur), `http_count` (nombre de liens), `diese` (nombre de hashtags), `nb_mots` (nombre de mots de la description).

La distribution des vues est très déséquilibrée : 2 chaînes représentent près de la moitié des vidéos, et les vidéos très vues (`log1p_views > 10`) sont peu représentées.

## Architecture

```
Miniature ──► ResNet50 (gelé, sans avg-pool) ──► 49 tokens × 2048 ─┐
                                                                  ├─► + encodage positionnel
Titre ──────► Llama-2-7B (8 bits, gelé) + Linear 4096→2048 ──────┘        sinusoïdal
                                                                  │
                                       Cross-attention bidirectionnelle (MDCAM)
                                       texte→image et image→texte, puis moyenne
                                                                  │
                                              Projecteur (Linear + ReLU + Dropout)
                                                                  │
   Description (Llama → projection 15 dim) ─┐                    │
   Embedding de chaîne (8 dim)               ├──── concaténation ─┘
   Année normalisée, http_count, diese, nb_mots ┘        │
                                                    Tête de régression
                                                         │
                                                  20 · sigmoid(x)  ──► log(1 + views)
```

Le modèle principal est `MultiModalAttentionRegressor` dans [`models/multimodalAttention.py`](models/multimodalAttention.py).

Choix d'entraînement :
- **Loss** : Huber (δ = 1), pour limiter l'influence des vidéos virales mal prédites.
- **Optimiseur** : AdamW + `ReduceLROnPlateau` (patience 3, facteur 0.5).
- **Validation temporelle** (`DataModuleTemporal`) : toutes les vidéos de 2023 + la moitié de celles de 2022 (~19.5 % des données), pour approcher la distribution du test.
- **Fine-tuning** en deux temps : entraînement initial, puis quelques epochs à faible learning rate sur les vidéos récentes.
- **Augmentation** des images à la volée (flou gaussien, rotation ≤ 10°).

Autres pistes explorées (voir le rapport) : DinoV2 seul, DinoV2 + DistilBERT, CLIP (`models/multimodal_CLIP.py`), classifieur pour router vers deux sous-modèles (`train_classifier.py`, abandonné), QLoRA.

## Organisation du dépôt

| Chemin | Contenu |
|---|---|
| `train.py` | Entraînement principal (Hydra + wandb) |
| `train_CLIP.py` | Entraînement de la baseline CLIP |
| `train_classifier.py` | Expérience classifieur / modèle mixte |
| `crossValidation.py` | Validation croisée (K-fold) |
| `create_submission.py`, `working_create_submission*.py` | Génération du fichier de soumission Kaggle |
| `val_loss.py` | Évaluation d'un checkpoint sur le set de validation |
| `models/` | Encodeurs (ResNet50, DinoV2, DistilBERT, Llama, CLIP…) et modèles multimodaux |
| `data/` | `Dataset` et `DataModule` (dont la séparation temporelle) |
| `configs/` | Configurations Hydra (modèle, dataset, optimiseur, loss) |
| `utils/` | Losses et outils de visualisation |
| `analyseDonnees.ipynb` | Analyse exploratoire des données |
| `analyseModele.ipynb` | Analyse du modèle (prédictions, poids d'attention, embeddings) |

## Utilisation

```bash
pip install -r requirements.txt
```

Llama-2 est un modèle à accès restreint sur Hugging Face : il faut avoir accepté la licence et être connecté (`huggingface-cli login`). L'encodeur charge Llama-2-7B quantifié en 8 bits (bitsandbytes) ; nous avons utilisé un GPU de 24 Go de VRAM.

Entraînement (les paramètres se trouvent dans `configs/train.yaml` et peuvent être surchargés en ligne de commande) :

```bash
python train.py
python train.py max_epochs=10 log=False
```

Création de la soumission :

```bash
python working_create_submission.py
```

⚠️ Les chemins des checkpoints sont codés en dur (`/Data/checkpoints/...` dans `configs/train.yaml` et dans les scripts de soumission) : il faut les adapter à votre machine.

## Résultats

- Le modèle prédit bien les vidéos à nombre de vues moyen (`log1p_views` entre 6 et 10) mais sous-estime les vidéos très vues, sous-représentées dans le dataset.
- Sur Kaggle, Y-VAMPX fait mieux que les baselines DinoV2 (image seule) et CLIP.
- Les études d'ablation montrent que l'année de publication est la variable la plus informative, suivie de l'image, de la description et du titre ; la chaîne et le nombre de liens pèsent peu.
- La cross-attention bidirectionnelle et l'encodage positionnel accélèrent la convergence et améliorent la loss de validation.

Le détail est dans le rapport et dans la présentation de soutenance.
