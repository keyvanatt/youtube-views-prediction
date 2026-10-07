# Y-VAMPX — YouTube Views Attention-based Multimodal Predictor

Project for the **CSC_43M04_EP** Data Challenge (Modal d'informatique, École polytechnique) — Louis Darrigol & Keyvan Attarian.

The goal is to predict the number of views of a YouTube video from its **thumbnail**, **title**, **description** and metadata (channel, year). The model predicts `log(1 + views)`.

## Data

- `dataset/train_val.csv`: ~15k videos (2011–2023), `dataset/test.csv`: ~3k videos (2024 – early 2025).
- Thumbnails (not versioned) are expected in `dataset/train_val/<id>.jpg` and `dataset/test/<id>.jpg`.
- Columns added to the original dataset: `channel_id` (channel encoded as an integer, 46 channels), `log1p_views`, `category` (used by the classifier), `http_count` (number of links), `diese` (number of hashtags), `nb_mots` (number of words in the description).

The view distribution is highly imbalanced: 2 channels account for almost half of the videos, and highly viewed videos (`log1p_views > 10`) are underrepresented.

## Architecture

```
Thumbnail ──► ResNet50 (frozen, no avg-pool) ──► 49 tokens × 2048 ─┐
                                                                  ├─► + sinusoidal
Title ──────► Llama-2-7B (8-bit, frozen) + Linear 4096→2048 ─────┘    positional encoding
                                                                  │
                                       Bidirectional cross-attention (MDCAM)
                                       text→image and image→text, then mean pooling
                                                                  │
                                              Projector (Linear + ReLU + Dropout)
                                                                  │
   Description (Llama → 15-dim projection) ──┐                   │
   Channel embedding (8 dim)                  ├─── concatenation ─┘
   Normalized year, http_count, diese, nb_mots ┘        │
                                                  Regression head
                                                        │
                                                 20 · sigmoid(x)  ──► log(1 + views)
```

The main model is `MultiModalAttentionRegressor` in [`models/multimodalAttention.py`](models/multimodalAttention.py).

Training choices:
- **Loss**: Huber (δ = 1), to limit the influence of poorly predicted viral videos.
- **Optimizer**: AdamW + `ReduceLROnPlateau` (patience 3, factor 0.5).
- **Temporal validation** (`DataModuleTemporal`): all 2023 videos + half of the 2022 videos (~19.5% of the data), to approximate the test distribution.
- **Two-stage fine-tuning**: initial training, then a few epochs at a low learning rate on recent videos.
- **On-the-fly image augmentation** (Gaussian blur, rotation ≤ 10°).

Other approaches explored (see the report): DinoV2 alone, DinoV2 + DistilBERT, CLIP (`models/multimodal_CLIP.py`), a classifier routing to two sub-models (`train_classifier.py`, abandoned), QLoRA.

## Repository structure

| Path | Content |
|---|---|
| `train.py` | Main training script (Hydra + wandb) |
| `train_CLIP.py` | CLIP baseline training |
| `train_classifier.py` | Classifier / mixed model experiment |
| `crossValidation.py` | K-fold cross-validation |
| `create_submission.py`, `working_create_submission*.py` | Kaggle submission generation |
| `val_loss.py` | Evaluates a checkpoint on the validation set |
| `models/` | Encoders (ResNet50, DinoV2, DistilBERT, Llama, CLIP…) and multimodal models |
| `data/` | `Dataset` and `DataModule` (including the temporal split) |
| `configs/` | Hydra configs (model, dataset, optimizer, loss) |
| `utils/` | Losses and visualization helpers |
| `analyseDonnees.ipynb` | Exploratory data analysis |
| `analyseModele.ipynb` | Model analysis (predictions, attention weights, embeddings) |

## Usage

```bash
pip install -r requirements.txt
```

Llama-2 is a gated model on Hugging Face: you need to accept its license and log in (`huggingface-cli login`). The encoder loads Llama-2-7B quantized to 8 bits (bitsandbytes); we used a GPU with 24 GB of VRAM.

Training (parameters are in `configs/train.yaml` and can be overridden from the command line):

```bash
python train.py
python train.py max_epochs=10 log=False
```

Creating a submission:

```bash
python working_create_submission.py
```

⚠️ Checkpoint paths are hard-coded (`/Data/checkpoints/...` in `configs/train.yaml` and in the submission scripts): adapt them to your machine.

## Results

- The model predicts mid-range videos well (`log1p_views` between 6 and 10) but underestimates highly viewed videos, which are underrepresented in the dataset.
- On Kaggle, Y-VAMPX outperforms the DinoV2 (thumbnail only) and CLIP baselines.
- Ablation studies show that the publication year is the most informative feature, followed by the image, the description and the title; the channel and the number of links matter little.
- Bidirectional cross-attention and positional encoding speed up convergence and improve the validation loss.

See the report and the defense slides for details.
