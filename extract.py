"""Phase 1: Extract embeddings from foundation models on RS datasets."""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import MODELS, DATASETS, EMBED_DIR, BATCH_SIZE, NUM_WORKERS
from utils import filter_zero_norm, l2_normalize


def _load_prithvi_model():
    """Load Prithvi-EO-1.0-100M from HuggingFace using its custom class."""
    import importlib.util
    from huggingface_hub import hf_hub_download

    repo_id = MODELS["prithvi"]["name"]
    mae_py = hf_hub_download(repo_id, "prithvi_mae.py")
    weights_path = hf_hub_download(repo_id, "Prithvi_EO_V1_100M.pt")

    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.PrithviMAE(
        img_size=224,
        patch_size=(1, 16, 16),
        num_frames=3,
        in_chans=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model


def extract_prithvi(dataset_name: str, dataset_cfg: dict) -> np.ndarray:
    """Extract embeddings from Prithvi-EO-1.0-100M."""
    import torchgeo.datasets as tgd

    model = _load_prithvi_model()
    model = model.cuda().eval()

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
            # Prithvi expects (B, C, T, H, W) with T=3 frames
            images = images.unsqueeze(2).expand(-1, -1, 3, -1, -1).cuda().float()

            # Forward through encoder, get list of layer features
            features = model.forward_features(images)
            # Use last layer output, mean-pool patch tokens
            emb = features[-1].mean(dim=1)

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
            # Select RGB bands, resize to 224x224, normalize for CLIP
            rgb_indices = dataset_cfg["rgb_bands"]
            images = images[:, rgb_indices, :, :]
            images = torch.nn.functional.interpolate(images.float(), size=(224, 224), mode="bilinear", align_corners=False)
            # Normalize to [0,1] then apply CLIP normalization
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.cuda()

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
