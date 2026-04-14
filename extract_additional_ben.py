"""Extract embeddings from additional models on BigEarthNet."""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import EMBED_DIR, BATCH_SIZE, NUM_WORKERS
from utils import filter_zero_norm, l2_normalize

DATA_ROOT = Path("data") / "bigearthnet"


def load_bigearthnet():
    """Load BigEarthNet dataset via torchgeo."""
    import torchgeo.datasets as tgd
    dataset = tgd.BigEarthNet(root=str(DATA_ROOT), bands="s2", download=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    return loader


def extract_dinov2_ben():
    """DINOv2 on BigEarthNet (RGB bands)."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model = model.cuda().eval()

    loader = load_bigearthnet()
    rgb_bands = [3, 2, 1]

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="DINOv2/bigearthnet"):
            images = batch["image"][:, rgb_bands, :, :]
            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False)
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.cuda()
            outputs = model(images)
            emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_georsclip_ben():
    """GeoRSCLIP (OpenAI ViT-L-14 fallback) on BigEarthNet (RGB)."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    model = model.cuda().eval()

    loader = load_bigearthnet()
    rgb_bands = [3, 2, 1]

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="GeoRSCLIP/bigearthnet"):
            images = batch["image"][:, rgb_bands, :, :]
            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False)
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.cuda()
            emb = model.encode_image(images)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_mae_base_ben():
    """MAE-base on BigEarthNet (RGB)."""
    from transformers import ViTMAEModel
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model = model.cuda().eval()

    loader = load_bigearthnet()
    rgb_bands = [3, 2, 1]

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="MAE-base/bigearthnet"):
            images = batch["image"][:, rgb_bands, :, :]
            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False)
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images - mean) / std
            images = images.cuda()
            outputs = model(images, noise=torch.zeros(images.shape[0], 196).cuda())
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_ssl4eo_ben():
    """SSL4EO MAE on BigEarthNet (all 12 S2 bands, padded to 13)."""
    import timm
    model = timm.create_model("vit_base_patch16_224", pretrained=False,
                               num_classes=0, in_chans=13)
    state_dict = torch.load("data/B13_vitb16_mae_ep99_enc.pth",
                            map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    loader = load_bigearthnet()

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="SSL4EO/bigearthnet"):
            images = batch["image"]  # 12 S2 bands
            if images.shape[1] < 13:
                pad = torch.zeros(images.shape[0], 13 - images.shape[1],
                                  images.shape[2], images.shape[3])
                images = torch.cat([images, pad], dim=1)
            elif images.shape[1] > 13:
                images = images[:, :13, :, :]
            images = torch.nn.functional.interpolate(
                images.float(), size=(224, 224), mode="bilinear", align_corners=False)
            images = images / (images.amax(dim=(2, 3), keepdim=True) + 1e-8)
            images = images.cuda()
            emb = model(images)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


MODELS = {
    "dinov2": extract_dinov2_ben,
    "georsclip": extract_georsclip_ben,
    "mae_base": extract_mae_base_ben,
    "ssl4eo": extract_ssl4eo_ben,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Extracting: {model_name} / bigearthnet")
        print(f"{'='*60}")

        try:
            raw = MODELS[model_name]()
            print(f"  Raw shape: {raw.shape}")

            filtered, mask = filter_zero_norm(raw)
            print(f"  After filtering: {filtered.shape}")

            normalized, norms = l2_normalize(filtered)

            out_path = EMBED_DIR / f"{model_name}_bigearthnet.npz"
            np.savez_compressed(out_path, embeddings=normalized, norms=norms, mask=mask)
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
