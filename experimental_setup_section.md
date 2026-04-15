# 3. Experimental Setup

## 3.1 Foundation Models

We select six foundation models spanning three training paradigms. The goal is not to benchmark the models themselves but to test whether the training objective, which shapes the geometry of the embedding space, affects quantization quality.

### Masked Autoencoder (MAE) Models

MAE models are trained to reconstruct masked patches from partial observations (He et al., 2022). The training signal is pixel-level reconstruction loss. There is no explicit pressure to distribute embeddings uniformly on the unit sphere.

**Prithvi-EO-1.0-100M** (Jakubik et al., 2023). A ViT-based Masked Autoencoder trained on multi-temporal Sentinel-2 imagery by IBM and NASA. It takes 6 multispectral bands (B2-B7) across 3 temporal frames as input. We extract embeddings by mean-pooling the encoder output from `forward_features()`, producing d=768 vectors. Prithvi is the most widely cited EO foundation model and represents the MAE paradigm applied specifically to remote sensing.

**MAE-base** (He et al., 2022). The original Masked Autoencoder (ViT-B/16) trained on ImageNet. We include it as a non-EO MAE baseline. It takes RGB input resized to 224x224. We extract embeddings by mean-pooling the encoder hidden states with no masking applied, producing d=768 vectors.

**SSL4EO** (Wang et al., 2023). A ViT-B/16 MAE trained on the SSL4EO-S12 dataset, which contains Sentinel-1 and Sentinel-2 imagery from 250K global locations. It takes 13 spectral bands as input. We use the pretrained encoder weights and extract d=768 vectors. SSL4EO represents a second EO-specific MAE model, trained on different data than Prithvi.

### Contrastive Models

Contrastive models are trained to pull matched pairs together and push unmatched pairs apart in the embedding space (Radford et al., 2021). The InfoNCE loss explicitly optimizes cosine similarity structure. Wang & Isola (2020) showed this produces embeddings with a "uniformity" property: they spread out on the unit sphere.

**RemoteCLIP ViT-B-32** (Liu et al., 2024). A CLIP model fine-tuned on remote sensing image-text pairs. It takes RGB input with CLIP preprocessing (resize to 224x224, CLIP normalization). We extract image embeddings via `encode_image()`, producing d=512 vectors. RemoteCLIP is specifically adapted for EO, making it the most relevant contrastive baseline.

**GeoRSCLIP ViT-L-14**. A CLIP ViT-L-14 with OpenAI pretrained weights (Radford et al., 2021). The GeoRSCLIP-specific weights (trained on the RS5M dataset) were unavailable at the time of our experiments. We use the OpenAI checkpoint, which is still contrastive (CLIP-trained) and produces d=768 vectors. This serves as a larger-scale contrastive model to test whether the pattern holds across model sizes.

### Self-Distillation Models

Self-distillation trains a student network to match the output of a momentum-updated teacher network (Caron et al., 2021). Unlike CLIP, there is no text supervision. Unlike MAE, there is no pixel reconstruction. The training signal is consistency between augmented views, which has been shown to produce highly uniform representations.

**DINOv2 ViT-B** (Oquab et al., 2024). Trained with self-distillation on a curated dataset of 142M images. We extract the CLS token from the final layer, producing d=768 vectors. DINOv2 is not trained on EO data, but it is widely used for remote sensing feature extraction due to its strong transfer properties. Including it tests whether self-distillation, a third training paradigm, produces the same isotropy benefits as contrastive learning.

### Summary

| Model | Training | Domain | Input | d |
|-------|----------|--------|-------|---|
| Prithvi | MAE | EO (Sentinel-2) | 6 MS bands, 3 frames | 768 |
| MAE-base | MAE | General (ImageNet) | RGB | 768 |
| SSL4EO | MAE | EO (Sentinel-1/2) | 13 bands | 768 |
| RemoteCLIP | Contrastive (CLIP) | EO (RS image-text) | RGB | 512 |
| GeoRSCLIP | Contrastive (CLIP) | General (WebImageText) | RGB | 768 |
| DINOv2 | Self-distillation | General (curated) | RGB | 768 |

## 3.2 Datasets

We use two standard remote sensing classification benchmarks repurposed for retrieval evaluation. Both are Sentinel-2 datasets with publicly available imagery and labels.

**EuroSAT** (Helber et al., 2019). 27,000 Sentinel-2 patches at 64x64 pixels across 10 land-use classes (industrial, residential, forest, river, etc.). We use the train split via TorchGeo (Stewart et al., 2022), yielding 16,200 patches. EuroSAT is small enough for rapid iteration and serves as our development benchmark.

**BigEarthNet-S2** (Sumbul et al., 2019). 590,326 Sentinel-2 patches at 120x120 pixels with 43 multi-label land-cover classes. We use the train split (269,695 patches) downloaded from Zenodo. BigEarthNet is 17x larger than EuroSAT and serves as our scale benchmark. It tests whether patterns observed on 16K vectors hold at 269K vectors.

Both datasets are loaded using TorchGeo (Stewart et al., 2022). For models requiring RGB input, we select bands B4, B3, B2 (true color). For Prithvi, we use bands B2-B7 (6 multispectral bands). For SSL4EO, we use all 13 available bands with zero-padding if fewer than 13 are available. All images are resized to 224x224 pixels via bilinear interpolation before being passed to the model.

### Dataset Statistics After Extraction

| Dataset | Patches | Prithvi dim | RemoteCLIP dim | Zero-norm filtered |
|---------|:----:|:----:|:----:|:----:|
| EuroSAT | 16,200 | 768 | 512 | 0 |
| BigEarthNet | 269,695 | 768 | 512 | 0 |

No zero-norm vectors were found in any model-dataset combination, indicating that all models produce non-degenerate embeddings for these datasets.

## 3.3 Hardware

All experiments were run on a single workstation with an NVIDIA RTX A3000 Laptop GPU (6 GB VRAM), 64 GB RAM, and an Intel Core i7 processor. Embedding extraction uses batch size 16 to fit within GPU memory. Quantization and search are performed on CPU using NumPy and FAISS.

Approximate runtimes:
- Embedding extraction: 2-3 minutes per model on EuroSAT, 50-90 minutes per model on BigEarthNet
- Full 9-method benchmark: 30 minutes on EuroSAT, 3-4 hours on BigEarthNet (bottleneck: adaptive Lloyd-Max and Hamming search loops)
- Ranking analysis (Kendall's tau): 30 minutes total across all models and datasets
