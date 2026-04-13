"""Extract embeddings from additional foundation models on EuroSAT."""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import EMBED_DIR, BATCH_SIZE, NUM_WORKERS
from utils import filter_zero_norm, l2_normalize

DATA_ROOT = Path("data") / "eurosat"


def load_eurosat_rgb():
    """Load EuroSAT dataset via torchgeo, return DataLoader."""
    import torchgeo.datasets as tgd
    dataset = tgd.EuroSAT(root=str(DATA_ROOT), download=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    return loader, [3, 2, 1]  # RGB band indices


def extract_dinov2():
    """Extract embeddings from DINOv2 ViT-B (facebook/dinov2-base)."""
    from transformers import AutoModel, AutoImageProcessor

    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.cuda().eval()

    loader, rgb_bands = load_eurosat_rgb()

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="DINOv2/eurosat"):
            images = batch["image"][:, rgb_bands, :, :]
            # Resize to 224x224 and normalize for DINOv2
            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False
            )
            # Normalize to [0, 1]
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.cuda()

            outputs = model(images)
            # CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_georsclip():
    """Extract embeddings from GeoRSCLIP ViT-L-14."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    # Try loading GeoRSCLIP weights
    try:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download("geolocal/GeoRSCLIP", "ViT-L-14.pt",
                                        local_dir="data")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("  Loaded GeoRSCLIP weights")
    except Exception as e:
        print(f"  Warning: Could not load GeoRSCLIP weights ({e})")
        print("  Using OpenAI ViT-L-14 pretrained weights as fallback")

    model = model.cuda().eval()

    loader, rgb_bands = load_eurosat_rgb()

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="GeoRSCLIP/eurosat"):
            images = batch["image"][:, rgb_bands, :, :]
            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False
            )
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.cuda()

            emb = model.encode_image(images)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_mae_base():
    """Extract embeddings from facebook/vit-mae-base (original MAE, ImageNet)."""
    from transformers import ViTMAEModel

    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model = model.cuda().eval()

    loader, rgb_bands = load_eurosat_rgb()

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="MAE-base/eurosat"):
            images = batch["image"][:, rgb_bands, :, :]
            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False
            )
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.cuda()

            outputs = model(images, noise=torch.zeros(images.shape[0], 196).cuda())
            # Mean pool all patch tokens (skip CLS if present)
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_ssl4eo():
    """Extract embeddings from SSL4EO-S12 MAE ViT-B/16 (RS-specific MAE)."""
    import timm

    # Create ViT-B/16 with 13-band input
    model = timm.create_model("vit_base_patch16_224", pretrained=False,
                               num_classes=0, in_chans=13)
    state_dict = torch.load("data/B13_vitb16_mae_ep99_enc.pth",
                            map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    # Load EuroSAT with all 13 bands
    import torchgeo.datasets as tgd
    dataset = tgd.EuroSAT(root=str(DATA_ROOT), download=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="SSL4EO/eurosat"):
            images = batch["image"]  # all bands
            # SSL4EO expects 13 bands; EuroSAT has 13 via torchgeo
            if images.shape[1] < 13:
                # Pad with zeros if fewer bands
                pad = torch.zeros(images.shape[0], 13 - images.shape[1],
                                  images.shape[2], images.shape[3])
                images = torch.cat([images, pad], dim=1)
            elif images.shape[1] > 13:
                images = images[:, :13, :, :]

            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False
            )
            # Normalize per-band to roughly [0, 1]
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            images = images.cuda()

            emb = model(images)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


MODELS = {
    "dinov2": {"extract_fn": extract_dinov2, "embed_dim": 768, "training": "self-distillation"},
    "georsclip": {"extract_fn": extract_georsclip, "embed_dim": 768, "training": "contrastive"},
    "mae_base": {"extract_fn": extract_mae_base, "embed_dim": 768, "training": "MAE"},
    "ssl4eo": {"extract_fn": extract_ssl4eo, "embed_dim": 768, "training": "MAE (RS)"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in models:
        cfg = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Extracting: {model_name} / eurosat")
        print(f"  Training type: {cfg['training']}")
        print(f"{'='*60}")

        try:
            raw = cfg["extract_fn"]()
            print(f"  Raw shape: {raw.shape}")

            filtered, mask = filter_zero_norm(raw)
            print(f"  After filtering: {filtered.shape}")

            normalized, norms = l2_normalize(filtered)

            out_path = EMBED_DIR / f"{model_name}_eurosat.npz"
            np.savez_compressed(out_path, embeddings=normalized, norms=norms, mask=mask)
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
