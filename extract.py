"""Phase 1: Extract embeddings from foundation models on RS datasets."""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import MODELS, DATASETS, EMBED_DIR, BATCH_SIZE, NUM_WORKERS
from utils import filter_zero_norm, l2_normalize


def extract_prithvi(dataset_name: str, dataset_cfg: dict) -> np.ndarray:
    """Extract embeddings from Prithvi-EO-1.0-100M."""
    from transformers import AutoModel
    import torchgeo.datasets as tgd

    model = AutoModel.from_pretrained(MODELS["prithvi"]["name"], trust_remote_code=True)
    model = model.cuda().eval()

    # Load dataset with appropriate bands
    # NOTE: adjust this based on actual torchgeo API for your dataset version
    if dataset_name == "bigearthnet":
        dataset = tgd.BigEarthNet(
            root=str(Path("data") / "bigearthnet"),
            bands="s2",
            download=True,
        )
    elif dataset_name == "eurosat":
        dataset = tgd.EuroSAT(
            root=str(Path("data") / "eurosat"),
            download=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Prithvi/{dataset_name}"):
            images = batch["image"]
            # Select Prithvi's 6 bands
            band_indices = dataset_cfg["prithvi_bands"]
            images = images[:, band_indices, :, :]
            images = images.cuda().float()

            # Prithvi is a ViT-MAE; use encoder output, take CLS token or mean pool
            outputs = model(images)
            # Adjust based on actual Prithvi output format:
            # Option A: CLS token
            # emb = outputs.last_hidden_state[:, 0, :]
            # Option B: Mean pool all patch tokens
            emb = outputs.last_hidden_state.mean(dim=1)

            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_remoteclip(dataset_name: str, dataset_cfg: dict) -> np.ndarray:
    """Extract embeddings from RemoteCLIP ViT-B-32."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    # Load RemoteCLIP weights
    state_dict = torch.load("data/RemoteCLIP-ViT-B-32.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    # Use torchgeo for loading, convert to RGB
    import torchgeo.datasets as tgd

    if dataset_name == "bigearthnet":
        dataset = tgd.BigEarthNet(root=str(Path("data") / "bigearthnet"), bands="s2", download=True)
    elif dataset_name == "eurosat":
        dataset = tgd.EuroSAT(root=str(Path("data") / "eurosat"), download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"RemoteCLIP/{dataset_name}"):
            images = batch["image"]
            # Select RGB bands and normalize for CLIP
            rgb_indices = dataset_cfg["rgb_bands"]
            images = images[:, rgb_indices, :, :]
            # TODO: apply CLIP preprocessing (resize, normalize)
            images = images.cuda().float()

            emb = model.encode_image(images)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract RS embeddings")
    parser.add_argument("--model", choices=["prithvi", "remoteclip", "all"], default="all")
    parser.add_argument("--dataset", choices=["bigearthnet", "eurosat", "all"], default="all")
    args = parser.parse_args()

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for model_name in models:
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"Extracting: {model_name} / {dataset_name}")
            print(f"{'='*60}")

            extract_fn = extract_prithvi if model_name == "prithvi" else extract_remoteclip

            raw_embeddings = extract_fn(dataset_name, DATASETS[dataset_name])
            print(f"  Raw shape: {raw_embeddings.shape}")

            # Filter zero-norm vectors (blank/cloudy tiles)
            filtered, mask = filter_zero_norm(raw_embeddings)
            print(f"  After filtering: {filtered.shape}")

            # L2 normalize
            normalized, norms = l2_normalize(filtered)

            # Save
            out_path = EMBED_DIR / f"{model_name}_{dataset_name}.npz"
            np.savez_compressed(
                out_path,
                embeddings=normalized,
                norms=norms,
                mask=mask,
            )
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
