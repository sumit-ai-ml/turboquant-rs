# TurboQuant RS

Benchmark for scalar vector quantization methods on remote sensing foundation model embeddings. Evaluates TurboQuant (rotation + Lloyd-Max optimal codebook) against training-free and data-adaptive baselines for approximate nearest neighbor retrieval.

## Methods

### Training-free (no data needed)
| Method | Description | Storage (d dims, b bits) |
|--------|-------------|--------------------------|
| **TurboQuant MSE** | Random rotation + Beta(d/2,d/2) optimal codebook | b*d/8 + 4 bytes |
| **RaBitQ** | Random rotation + binarization (Gao & Long, SIGMOD 2024) | d/8 bytes |
| Binary Hash | sign(x) + Hamming distance | d/8 bytes |
| SimHash Multi-bit | k random hyperplanes, 1 bit each | k/8 bytes |
| Uniform SQ | Random rotation + uniform grid | b*d/8 + 4 bytes |
| FlyHash | Sparse random expansion + winner-take-all | k/8 bytes |
| RandProj Quant | Random Gaussian projection + 8-bit quantization | m bytes |

### Requires training data
| Method | Description | Storage |
|--------|-------------|---------|
| **Product Quantization** | FAISS PQ with learned subspace codebooks | m*nbits/8 bytes |
| TurboQuant Adaptive | Rotation + empirical Lloyd-Max codebook | b*d/8 + 4 bytes |

## Results (EuroSAT, R@10, 4-bit budget)

### Prithvi (d=768)
| Method | R@10 | B/vec | Training? |
|--------|------|-------|-----------|
| FP32 Exact | 1.000 | 3072 | - |
| Product Quant | 0.961 | 384 | Yes |
| TurboQuant MSE | 0.779 | 388 | **No** |
| TurboQuant Adaptive | 0.782 | 388 | Yes |
| SimHash Multi-bit | 0.702 | 384 | No |
| Uniform SQ | 0.502 | 388 | No |
| RaBitQ (1-bit) | 0.502 | 96 | No |
| Binary Hash (1-bit) | 0.451 | 96 | No |
| FlyHash | 0.468 | 384 | No |

### RemoteCLIP (d=512)
| Method | R@10 | B/vec | Training? |
|--------|------|-------|-----------|
| FP32 Exact | 1.000 | 2048 | - |
| Product Quant | 0.961 | 256 | Yes |
| TurboQuant MSE | 0.911 | 260 | **No** |
| TurboQuant Adaptive | 0.912 | 260 | Yes |
| SimHash Multi-bit | 0.751 | 256 | No |
| RandProj Quant | 0.718 | 256 | No |
| RaBitQ (1-bit) | 0.567 | 64 | No |
| Binary Hash (1-bit) | 0.607 | 64 | No |

**TurboQuant MSE is the best training-free method**, closing 60-70% of the gap between naive hashing and data-adaptive PQ without requiring any training data.

## Models

- **Prithvi-EO-1.0-100M** (IBM/NASA): ViT-MAE for multispectral Sentinel-2, embed_dim=768
- **RemoteCLIP ViT-B-32**: CLIP-based vision encoder for RGB remote sensing, embed_dim=512

## Datasets

- **EuroSAT**: 27,000 Sentinel-2 patches, 10 land-use classes
- **BigEarthNet-S2**: 590,326 Sentinel-2 patches, 43 multi-label classes (requires ~66GB download)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
bash setup.sh
```

Requires Python 3.10+, PyTorch 2.0+ with CUDA. Tested on NVIDIA RTX A3000 (6GB).

## Running

### Full pipeline
```bash
bash run.sh
```

### Step by step
```bash
# Phase 0: Sanity check (synthetic data, no GPU needed)
python sanity_check.py

# Phase 1: Extract embeddings
python extract.py --model prithvi --dataset eurosat
python extract.py --model remoteclip --dataset eurosat

# Phase 2: Validate Beta distribution assumption
python validate.py --model all --dataset eurosat

# Phase 3: Benchmark all methods
python benchmark.py --model all --dataset eurosat --method all

# Phase 4: Generate tables and figures
python analyze.py
```

### Run individual methods
```bash
python benchmark.py --method turboquant_mse --bits 2 3 4
python benchmark.py --method product_quant --bits 4 --seeds 42
python benchmark.py --method rabitq  # 1-bit only, ignores --bits
```

Results go to `results/`, figures to `figures/`.

## Key Findings

1. **TurboQuant MSE is the best training-free quantizer** for RS embedding retrieval, significantly ahead of SimHash, RaBitQ, and other training-free alternatives.

2. **The Beta(d/2,d/2) codebook matters**: Uniform scalar quantization with the same rotation gets R@10=0.502 vs TurboQuant's 0.779 on Prithvi. The analytically-optimal codebook accounts for most of the gain.

3. **The Beta assumption is violated** on real RS embeddings (KS test D=0.4-0.6), yet TurboQuant still works well. Data-adaptive codebooks give only marginal improvement (+0.3% R@10).

4. **QJL residual correction hurts** for cosine-similarity retrieval. The sign sketch variance dominates the correction signal. Multiple independent implementations reached the same conclusion.

5. **RaBitQ's advantage is the rotation, not the correction factor**. At 1-bit, RaBitQ beats binary hash on structured embeddings (Prithvi: +5% R@10) but not on already-isotropic ones (RemoteCLIP: -4%).

6. **PQ still wins on recall** when training data is available (R@10=0.96 vs 0.78-0.91 for TurboQuant). TurboQuant's value is zero-training instant quantization at scale.

## Project Structure

```
quantize.py      # All quantization methods (TQ, PQ, RaBitQ, baselines)
benchmark.py     # Phase 3: run recall benchmarks
extract.py       # Phase 1: extract embeddings from foundation models
validate.py      # Phase 2: test Beta distribution assumption
analyze.py       # Phase 4: tables, figures, scaling projections
sanity_check.py  # Phase 0: verify implementation on synthetic data
config.py        # Experiment configuration
utils.py         # Rotation matrices, normalization, metrics
setup.sh         # Install dependencies
run.sh           # Run full pipeline
```

## References

- **TurboQuant**: Guo et al., "TurboQuant: Online Vector Quantization with Near-Optimal Distortion", 2024
- **RaBitQ**: Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search", SIGMOD 2024
- **Prithvi**: Jakubik et al., "Foundation Models for Generalist Geospatial Artificial Intelligence", 2023
- **RemoteCLIP**: Liu et al., "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing", 2023
- **EuroSAT**: Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification", 2019
- **BigEarthNet**: Sumbul et al., "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding", 2019
