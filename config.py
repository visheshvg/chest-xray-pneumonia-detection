"""All configuration values in one place."""
import os
import torch

# === Dataset paths ===
DATASET_DIR = "chest_xray"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")
TEST_DIR  = os.path.join(DATASET_DIR, "test")

# === Image sizes (per model family) ===
IMG_SIZE_DEFAULT          = 224   # Custom CNN, DenseNet121, ResNet50
IMG_SIZE_EFFICIENTNET_B4  = 380   # EfficientNetB4

# === Hyperparameters ===
BATCH_SIZE      = 32      # 32 fits comfortably in 8GB VRAM at 224. Drop to 16 for B4.
BATCH_SIZE_B4   = 16
BATCH_SIZE_MEDMAMBA = 5   # smaller batch = less VRAM, same accuracy
EPOCHS_CUSTOM   = 25      # custom CNN, single phase
EPOCHS_PHASE1   = 10      # pretrained: feature extraction
EPOCHS_PHASE2   = 15      # pretrained: fine-tuning
LEARNING_RATE   = 1e-4
FINETUNE_LR     = 1e-5
NUM_WORKERS     = 4       # DataLoader workers; lower to 2 if Windows complains

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True            # mixed precision — speeds up training on RTX 3070

# === Classes ===
CLASSES     = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 1                          # binary => sigmoid

# === Reproducibility ===
RANDOM_SEED = 42

# === Output folders ===
OUTPUT_DIR  = "outputs"
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for d in (WEIGHTS_DIR, PLOTS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
