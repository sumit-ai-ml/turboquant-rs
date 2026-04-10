"""Experiment configuration for TurboQuant RS benchmark."""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EMBED_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# --- Models ---
MODELS = {
    "prithvi": {
        "name": "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
        "embed_dim": 768,
        "input_bands": "multispectral",  # 6 bands: B2,B3,B4,B5,B6,B7
    },
    "remoteclip": {
        "name": "ViT-B-32",
        "embed_dim": 512,
        "input_bands": "rgb",
        "weights_url": "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt",
    },
}

# --- Datasets ---
DATASETS = {
    "bigearthnet": {
        "name": "BigEarthNet-S2",
        "num_patches": 590_326,
        "num_classes": 43,
        "prithvi_bands": [1, 2, 3, 4, 5, 6],  # B2-B7 indices in BigEarthNet
        "rgb_bands": [3, 2, 1],  # B4, B3, B2 for true color
    },
    "eurosat": {
        "name": "EuroSAT",
        "num_patches": 27_000,
        "num_classes": 10,
        "prithvi_bands": [1, 2, 3, 4, 5, 6],
        "rgb_bands": [3, 2, 1],
    },
}

# --- Quantization ---
BITS = [2, 3, 4]
SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds for error bars
TRAIN_SPLIT = 0.8  # 80% for PQ codebook training, 20% for evaluation

# --- Methods ---
METHODS = [
    "fp32_exact",       # upper bound, no compression
    "binary_hash",      # sign(x) -> Hamming distance
    "product_quant",    # FAISS PQ (trained on 80% split)
    "turboquant_mse",   # rotation + Lloyd-Max optimal codebook
    "turboquant_prod",  # MSE + QJL residual correction
]

# --- Evaluation ---
RECALL_K = [1, 10, 100]

# --- Hardware ---
BATCH_SIZE = 16  # conservative for A3000 6GB VRAM
NUM_WORKERS = 4
