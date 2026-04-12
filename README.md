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

## Results (R@10, 4-bit budget unless noted)

### EuroSAT (16K vectors)

| Method | Prithvi (d=768) | RemoteCLIP (d=512) | B/vec | Training? |
|--------|----------------|-------------------|-------|-----------|
| FP32 Exact | 1.000 | 1.000 | 3072 / 2048 | - |
| Product Quant | 0.961 | 0.961 | 384 / 256 | Yes |
| **TurboQuant MSE** | **0.779** | **0.911** | 388 / 260 | **No** |
| TurboQuant Adaptive | 0.782 | 0.912 | 388 / 260 | Yes |
| SimHash Multi-bit | 0.702 | 0.751 | 384 / 256 | No |
| RandProj Quant | 0.394 | 0.718 | 384 / 256 | No |
| Uniform SQ | 0.502 | 0.549 | 388 / 260 | No |
| FlyHash | 0.468 | 0.545 | 384 / 256 | No |
| RaBitQ (1-bit) | 0.502 | 0.567 | 96 / 64 | No |
| Binary Hash (1-bit) | 0.451 | 0.607 | 96 / 64 | No |

### BigEarthNet (269K vectors)

| Method | Prithvi (d=768) | RemoteCLIP (d=512) | B/vec | Training? |
|--------|----------------|-------------------|-------|-----------|
| FP32 Exact | 1.000 | 1.000 | 3072 / 2048 | - |
| Product Quant | 0.925 | 0.944 | 384 / 256 | Yes |
| **TurboQuant MSE** | **0.572** | **0.878** | 388 / 260 | **No** |
| TurboQuant Adaptive | 0.584 | 0.887 | 388 / 260 | Yes |
| SimHash Multi-bit | 0.481 | 0.648 | 384 / 256 | No |
| RandProj Quant | 0.073 | 0.619 | 384 / 256 | No |
| Uniform SQ | 0.255 | 0.399 | 388 / 260 | No |
| FlyHash | 0.207 | 0.409 | 384 / 256 | No |
| RaBitQ (1-bit) | 0.256 | 0.418 | 96 / 64 | No |
| Binary Hash (1-bit) | 0.273 | 0.473 | 96 / 64 | No |

### Gap closed by TurboQuant MSE (training-free, between Binary Hash and PQ)

| Setting | Gap Closed |
|---------|-----------|
| RemoteCLIP / BigEarthNet | **86%** |
| RemoteCLIP / EuroSAT | **86%** |
| Prithvi / EuroSAT | **64%** |
| Prithvi / BigEarthNet | **46%** |

**TurboQuant MSE is the best training-free method**, closing 46-86% of the gap between naive hashing and data-adaptive PQ without requiring any training data.

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

## Reproducing Results

### Quick start (EuroSAT only)
```bash
bash run.sh
```

### Full reproduction (all datasets)

#### Phase 0: Sanity check (synthetic data, no GPU needed)
```bash
python sanity_check.py
```
Validates the quantization implementation on synthetic vectors (isotropic, clustered, tight clusters). Checks rotation quality, codebook range, and recall monotonicity.

#### Phase 1: Extract embeddings
```bash
# EuroSAT (~27K patches, ~2 min each)
python extract.py --model prithvi --dataset eurosat
python extract.py --model remoteclip --dataset eurosat

# BigEarthNet (~269K patches used for train split, ~50 min each)
# Requires downloading BigEarthNet-S2 from Zenodo (~66GB)
python extract.py --model prithvi --dataset bigearthnet
python extract.py --model remoteclip --dataset bigearthnet
```
Runs each foundation model on each dataset. Prithvi uses the custom `PrithviMAE` class loaded from HuggingFace (`ibm-nasa-geospatial/Prithvi-EO-1.0-100M`). RemoteCLIP uses `open_clip` ViT-B-32 with weights from `chendelong/RemoteCLIP`. Embeddings are L2-normalized and saved to `embeddings/*.npz`.

#### Phase 2: Validate Beta distribution assumption
```bash
python validate.py --model all --dataset eurosat
```
KS-tests each rotated coordinate against Beta(d/2, d/2). Generates QQ plots in `figures/`.

#### Phase 3: Benchmark all methods
```bash
# EuroSAT (~30 min)
python benchmark.py --model all --dataset eurosat --method all

# BigEarthNet (~3-4 hours, bottleneck: adaptive Lloyd-Max + Hamming loops)
python benchmark.py --model all --dataset bigearthnet --method all
```
For each (model, dataset, method, bits, seed) configuration:
1. Split embeddings 80/20 into train/eval
2. Use first min(1000, 10% of eval) vectors as queries, rest as database
3. Encode queries + database with each method
4. Compute ground-truth top-k via FP32 inner product
5. Compute approximate top-k via the method's search
6. Measure Recall@1/10/100 across 5 seeds (42, 123, 456, 789, 1024)

RaBitQ is 1-bit only and skips the bits parameter loop automatically.

#### Phase 4: Generate paper assets
```bash
python analyze.py                  # Summary tables and basic plots
python generate_paper_assets.py    # All figures (PNG+PDF), LaTeX tables, CSV
```
Merges all results, aggregates over seeds (mean +/- std), generates:
- `figures/fig1_recall_vs_bits.{png,pdf}` -- Main figure: R@10 vs bits (2x2 grid)
- `figures/fig2_pareto.{png,pdf}` -- Compression ratio vs recall (Pareto frontier)
- `figures/fig3_training_free_comparison.{png,pdf}` -- Bar chart: all training-free methods
- `figures/fig4_scaling.{png,pdf}` -- EuroSAT (16K) vs BigEarthNet (269K) scaling
- `figures/fig5_codebook_ablation.{png,pdf}` -- Beta vs Adaptive vs Uniform codebook
- `figures/qq_*.png` -- QQ plots for Beta distribution validation
- `results/table_main.tex` -- LaTeX tables sorted by R@10
- `results/all_results.csv` -- Complete results for custom analysis
- `results/aggregated_results.json` -- Aggregated JSON

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
