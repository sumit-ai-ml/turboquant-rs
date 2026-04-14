# TurboQuant RS

Training-free embedding compression for remote sensing retrieval. Benchmarks TurboQuant against 8 other quantization methods across 6 foundation models and 2 datasets.

**Main finding:** TurboQuant's retrieval recall depends on the coordinate independence of the embedding distribution (Pearson r = -0.951). Contrastive and self-distillation models compress well. MAE models don't.

## Results

### 6-Model Isotropy Test

**EuroSAT (16K vectors), TurboQuant MSE 4-bit R@10:**

| Model | Training | Coord Corr | TQ MSE | PQ | Binary Hash | Gap Closed |
|-------|----------|:----:|:----:|:----:|:----:|:----:|
| DINOv2 | Self-distillation | 0.132 | **0.943** | 0.960 | 0.654 | 95% |
| RemoteCLIP | Contrastive (CLIP) | 0.205 | **0.911** | 0.961 | 0.607 | 86% |
| GeoRSCLIP | Contrastive (CLIP) | 0.190 | **0.882** | 0.965 | 0.576 | 79% |
| MAE-base | MAE (ImageNet) | 0.510 | 0.859 | 0.953 | 0.179 | 88% |
| SSL4EO | MAE (RS) | 0.293 | 0.834 | 0.968 | 0.609 | 62% |
| Prithvi | MAE (RS) | 0.629 | 0.779 | 0.961 | 0.451 | 64% |

Pearson r(coord correlation, TQ R@10) = **-0.851**

**BigEarthNet (269K vectors), TurboQuant MSE 4-bit R@10:**

| Model | Training | Coord Corr | TQ MSE | PQ | Binary Hash | Gap Closed |
|-------|----------|:----:|:----:|:----:|:----:|:----:|
| DINOv2 | Self-distillation | 0.253 | **0.900** | 0.947 | 0.483 | 90% |
| RemoteCLIP | Contrastive (CLIP) | 0.215 | **0.878** | 0.944 | 0.473 | 86% |
| GeoRSCLIP | Contrastive (CLIP) | 0.247 | **0.830** | 0.950 | 0.447 | 76% |
| SSL4EO | MAE (RS) | 0.345 | 0.770 | 0.955 | 0.468 | 62% |
| MAE-base | MAE (ImageNet) | 0.521 | 0.737 | 0.935 | 0.128 | 76% |
| Prithvi | MAE (RS) | 0.663 | 0.572 | 0.925 | 0.273 | 46% |

Pearson r(coord correlation, TQ R@10) = **-0.951**

### All Methods Comparison (BigEarthNet, Prithvi + RemoteCLIP)

| Method | Bits | Prithvi R@10 | RemoteCLIP R@10 | B/vec | Training? |
|--------|:----:|:----:|:----:|:----:|:-:|
| FP32 Exact | - | 1.000 | 1.000 | 3072 / 2048 | - |
| Product Quant | 4 | 0.925 | 0.944 | 384 / 256 | Yes |
| **TurboQuant MSE** | **4** | **0.572** | **0.878** | **388 / 260** | **No** |
| TurboQuant Adaptive | 4 | 0.584 | 0.887 | 388 / 260 | Yes |
| SimHash Multi-bit | 4 | 0.481 | 0.648 | 384 / 256 | No |
| RandProj Quant | 4 | 0.073 | 0.619 | 384 / 256 | No |
| Uniform SQ | 4 | 0.255 | 0.399 | 388 / 260 | No |
| FlyHash | 4 | 0.207 | 0.409 | 384 / 256 | No |
| RaBitQ | 1 | 0.256 | 0.418 | 96 / 64 | No |
| Binary Hash | 1 | 0.273 | 0.473 | 96 / 64 | No |

### Codebook Ablation (BigEarthNet)

| Method | Prithvi R@10 | RemoteCLIP R@10 |
|--------|:----:|:----:|
| TQ MSE (Beta codebook) | 0.572 | 0.878 |
| TQ Adaptive (empirical codebook) | 0.584 | 0.887 |
| Uniform SQ (no codebook) | 0.255 | 0.399 |

Beta codebook gives 2.2x recall over uniform. Adaptive codebook adds only +1%.

### Scaling: EuroSAT (16K) to BigEarthNet (269K)

| Model | EuroSAT | BigEarthNet | Drop |
|-------|:----:|:----:|:----:|
| DINOv2 | 0.943 | 0.900 | -0.043 |
| RemoteCLIP | 0.911 | 0.878 | -0.033 |
| GeoRSCLIP | 0.882 | 0.830 | -0.052 |
| SSL4EO | 0.834 | 0.770 | -0.064 |
| MAE-base | 0.859 | 0.737 | -0.122 |
| Prithvi | 0.779 | 0.572 | -0.207 |

Isotropic models degrade 3-6%. Anisotropic models degrade 12-21%.

## Models

**MAE / Reconstruction:**
- **Prithvi-EO-1.0-100M** (Jakubik et al., 2023): ViT-MAE on Sentinel-2, d=768
- **MAE-base** (He et al., 2022): ViT-MAE on ImageNet, d=768
- **SSL4EO** (Wang et al., 2022): ViT-B/16 MAE on Sentinel-1/2, d=768

**Contrastive:**
- **RemoteCLIP ViT-B-32** (Liu et al., 2024): CLIP on RS image-text pairs, d=512
- **GeoRSCLIP ViT-L-14**: CLIP ViT-L-14 (OpenAI pretrained), d=768

**Self-distillation:**
- **DINOv2 ViT-B** (Oquab et al., 2024): Self-distillation, d=768

## Datasets

- **EuroSAT**: 16,200 Sentinel-2 patches, 10 land-use classes, 64x64 pixels
- **BigEarthNet-S2**: 269,695 Sentinel-2 patches, 43 multi-label classes, 120x120 pixels

## Methods

### Training-free
| Method | Description | Storage |
|--------|-------------|---------|
| **TurboQuant MSE** | Random rotation + Beta(d/2,d/2) optimal codebook | b*d/8 + 4 bytes |
| RaBitQ | Random rotation + binarization + Hamming | d/8 bytes |
| Binary Hash | sign(x) + Hamming distance | d/8 bytes |
| SimHash Multi-bit | k random hyperplanes, 1 bit each | k/8 bytes |
| Uniform SQ | Random rotation + uniform grid [-1,1] | b*d/8 + 4 bytes |
| FlyHash | Sparse random expansion + winner-take-all | k/8 bytes |
| RandProj Quant | Random Gaussian projection + 8-bit quantization | m bytes |

### Requires training
| Method | Description | Storage |
|--------|-------------|---------|
| **Product Quantization** | FAISS PQ with learned codebooks | m*nbits/8 bytes |
| TurboQuant Adaptive | Rotation + empirical Lloyd-Max codebook | b*d/8 + 4 bytes |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
bash setup.sh
```

Requires Python 3.10+, PyTorch 2.0+ with CUDA. Tested on NVIDIA RTX A3000 (6GB).

## Reproducing Results

### Quick start (EuroSAT, 2 original models)
```bash
bash run.sh
```

### Full reproduction

**Phase 0 — Sanity check** (no GPU needed):
```bash
python sanity_check.py
```

**Phase 1 — Extract embeddings:**
```bash
# Original 2 models (Prithvi + RemoteCLIP)
python extract.py --model prithvi --dataset eurosat
python extract.py --model remoteclip --dataset eurosat
python extract.py --model prithvi --dataset bigearthnet
python extract.py --model remoteclip --dataset bigearthnet

# Additional 4 models (DINOv2, GeoRSCLIP, MAE-base, SSL4EO)
python extract_additional.py --model all          # EuroSAT
python extract_additional_ben.py --model all       # BigEarthNet (~2hr/model)
```

**Phase 2 — Validate Beta assumption:**
```bash
python validate.py --model all --dataset eurosat
```

**Phase 3 — Benchmark:**
```bash
# Full 9-method sweep (original 2 models)
python benchmark.py --model all --dataset eurosat --method all
python benchmark.py --model all --dataset bigearthnet --method all

# 6-model isotropy test (TQ + PQ + BinHash only)
# Results generated by inline scripts — see generate_paper_assets_v2.py
```

**Phase 4 — Generate paper assets:**
```bash
python generate_paper_assets_v2.py
```

Generates 6 figures (PNG+PDF), LaTeX tables, and CSV exports in `figures/` and `results/`.

### Run individual methods
```bash
python benchmark.py --method turboquant_mse --bits 2 3 4
python benchmark.py --method product_quant --bits 4 --seeds 42
python benchmark.py --method rabitq
```

## Key Findings

1. **Coordinate independence predicts TQ recall** with r = -0.951 across 6 models. This is a stronger predictor than KS D statistic (r = -0.507) or training paradigm labels.

2. **TurboQuant MSE is the best training-free method** across all 6 models and both datasets, 9-23% ahead of SimHash (runner-up).

3. **The Beta codebook is essential.** Uniform quantization with the same rotation gives 2.2x worse recall. But a data-adaptive codebook gives only +1% over the fixed Beta codebook.

4. **DINOv2 (self-distillation) compresses best**, even better than CLIP models. Self-distillation produces the most isotropic embeddings.

5. **The model choice matters as much as the quantizer choice.** Switching from Prithvi to DINOv2 improves TQ R@10 from 0.572 to 0.900 on BigEarthNet. That's a bigger gain than switching from TQ to PQ.

6. **QJL correction hurts.** TurboQuant's "Prod" variant degrades recall for cosine retrieval.

7. **Uniform SQ is insensitive to bits** at high d. At d=768, rotated coordinates live in +/-0.036. A [-1,1] grid wastes 96% of bins.

## Project Structure

```
quantize.py                 # All 9 quantization methods
benchmark.py                # Phase 3: recall benchmarks
extract.py                  # Phase 1: Prithvi + RemoteCLIP extraction
extract_additional.py       # Phase 1: DINOv2, GeoRSCLIP, MAE-base, SSL4EO (EuroSAT)
extract_additional_ben.py   # Phase 1: same 4 models on BigEarthNet
validate.py                 # Phase 2: Beta distribution validation
analyze.py                  # Phase 4: summary tables and plots
generate_paper_assets_v2.py # Phase 4: all paper figures, LaTeX, CSV
sanity_check.py             # Phase 0: synthetic data validation
config.py                   # Experiment configuration
utils.py                    # Rotation matrices, normalization, metrics
setup.sh                    # Install dependencies
run.sh                      # Quick pipeline (EuroSAT only)
paper.md                    # Full paper draft
```

## Outputs

```
figures/
  fig1_correlation_vs_recall.{png,pdf}  # Coord corr vs TQ R@10 (the money plot)
  fig2_recall_vs_bits.{png,pdf}         # R@10 vs bits per model
  fig3_training_free_methods.{png,pdf}  # All training-free methods compared
  fig4_codebook_ablation.{png,pdf}      # Beta vs Adaptive vs Uniform
  fig5_six_model_bars.{png,pdf}         # 6-model grouped bar chart
  fig6_scaling.{png,pdf}                # EuroSAT vs BigEarthNet scaling
  qq_*.png                              # QQ plots for Beta validation

results/
  six_model_results.json                # EuroSAT 6-model results
  six_model_ben_results.json            # BigEarthNet 6-model results
  benchmark_results.json                # Full 9-method benchmark (all seeds)
  rabitq_results.json                   # RaBitQ results
  table_6model.tex                      # LaTeX tables for paper
  paper_results.csv                     # CSV export for custom analysis
```

## References

- **TurboQuant**: Guo et al., "TurboQuant: Online Vector Quantization with Near-Optimal Distortion", arXiv:2501.06036, 2024
- **RaBitQ**: Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound", SIGMOD 2024
- **Product Quantization**: Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
- **SimHash**: Charikar, "Similarity Estimation Techniques from Rounding Algorithms", STOC 2002
- **FlyHash**: Dasgupta et al., "A Neural Algorithm for a Fundamental Computing Problem", Science 2017
- **Prithvi**: Jakubik et al., "Foundation Models for Generalist Geospatial Artificial Intelligence", 2023
- **RemoteCLIP**: Liu et al., "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing", IEEE TGRS 2024
- **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024
- **SSL4EO**: Wang et al., "SSL4EO-S12: A Large-Scale Multi-Modal Dataset for Self-Supervised Learning in EO", IEEE GRSM 2023
- **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
- **EuroSAT**: Helber et al., "EuroSAT: A Novel Dataset for Land Use Classification", IEEE JSTARS 2019
- **BigEarthNet**: Sumbul et al., "BigEarthNet: A Large-Scale Benchmark Archive", IEEE IGARSS 2019
- **Uniformity**: Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity", ICML 2020
