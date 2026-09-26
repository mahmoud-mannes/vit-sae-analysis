# vit-sae-analysis

A mechanistic study of how absolute and rotary positional encodings shape
spatial structure inside pretrained Vision Transformers; what carries the
positional signal, where it lives, how causally load-bearing it is, and why
the two encoding schemes turn out to require different instruments to study
at all.

This repository extends Mannes, *Positional Encodings Anchor Spatial
Structure in Vision Transformers: A Geometric Perspective on Robustness*
(arXiv 2606.00124), moving from a from-scratch ViT-S study to real pretrained
ViT-Base checkpoints, and from representational geometry to direct causal
intervention: first via sparse autoencoders, then via an exact intervention
on the rotary basis itself once the SAE route hit a real instrument problem
(see "The reconstruction confound," below).

## Team

- **Mahmoud Mannes** First-author, conducting mechanistic interpretability research for over a year, with a focus on the internal representations of Vision Transformers.

- **Aravind Kannappan:** Recently graduated from NYU with an MS. Previously a Research Fellow at EleutherAI and SPAR working on mechanistic interpretability.

- **Nikhil Maturi:** Working on mitigating AI x-risk through work on interpretability/alignment/control, and improving human health through research in BioML. Research Fellow at EleutherAI as of September 2026.

- **Jiwon Jeong:** M.S. in Artificial Intelligence, working mainly on NLP and LLMs. Previous research on LLM reasoning and commonsense QA, more recently on Transformer architecture and internals. Interested in APE vs. RoPE and how positional and spatial representations are formed inside ViTs.

- **Parva Mehta:** Working on mechanistic interpretability of transformer internals, with papers in the NeurIPS Workshop pipeline. Familiar with ViT internals, probing, and ablation style causal work. Also working on a computer vision patent related project and an upcoming paper on distilled ViTs.

## Checkpoint naming for APE in the repository

| Short name                         | Checkpoint                         |
| ---------------------------------  | ---------------------------------- |
| `Supervised ViT-B` or `first_seed` | `google/vit-base-patch16-224`      |
| `DINO ViT-B` or `second_seed`      | `vit_base_patch16_224.dino`        |
| `AugReg ViT-B` or `third_seed`     | `vit_base_patch16_224.augreg_in1k` |
| `SAM ViT-B` or `fourth_seed`       | `vit_base_patch16_224.sam_in1k`    |

Some result directories use names such as `first_seed`, `second_seed`, `third_seed`, and `fourth_seed`. These names **do not denote different random seeds or independently trained models**.

They are simply labels for different pretrained checkpoints used in the corresponding analyses. The terminology originated as convenient internal naming during development and is retained in some result paths for compatibility with the existing experiment artifacts.

Where relevant, the checkpoint identities should therefore be interpreted according to the model mapping above rather than as independent random-seed runs.

| Role | Model | Position encoding | Source |
| --- | --- | --- | --- |
| APE | `google/vit-base-patch16-224` | learned absolute | transformers |
| RoPE | `vit_base_patch16_rope_224.naver_in1k` | rotary | timm |
| APE (replication) | DINOv1 ViT-B/16 | learned absolute | timm |
| RoPE (replication) | DINOv3 ViT-B/16 | rotary | timm |


## Core methodology

**SSDC (Spatial Similarity Distance Correlation).** Spearman rank correlation
between token-pair cosine similarity and negative spatial distance. High SSDC
= spatially near tokens are represented similarly.

**RPI (Random Permutation at Inference).** Shuffle token order immediately after image patching while pinning
positional signal to sequence index, to isolate index-anchored structure from
content-driven structure.

**RPI for linear probes — a separate justification from RPI for SSDC.**
SSDC is a relational metric, so RPI's job there is to decorrelate
content-adjacency from index-adjacency across the whole similarity matrix.
For a per-example linear probe, RPI does something related but distinct: it
puts genuinely novel content–position pairings in front of the model, which
is exactly what's needed to test whether decodable position is caused by the
positional-encoding mechanism itself rather than by natural image priors
(sky-is-up-style shortcuts).

**Fragility.** `fragility = 1 - shifted_acc / baseline_acc` under distribution
shift (currently unused in main paper)

## Sparse autoencoders

In-depth discussion of SAE implementation and results can be found in the docs directory.

## Repository structure

### `docs/`

Documentation for the experimental methodology and results.

### `project_code/src/`

Main implementation.

* `main/` — model loading, data preparation, and corruption datasets.
* `metrics/` — SSDC, positional probes, effective-rank, and robustness metrics.
* `interventions/` — generic activation ablations and corruption utilities.
* `experiments/` — experiment entry points and analysis pipelines, including APE/RoPE causal studies, activation patching, layerwise ablations, and replication experiments.
* `SAE/` — sparse autoencoder training, activation extraction, storage, and evaluation.
* `SAE_feature_analysis/` — feature selection, positional selectivity analysis, and feature visualization.
* `SAE_causal/` — SAE-based causal feature ablations and associated hooks.

### `notebooks/`

A standalone notebook demonstrating the SSDC/ablation workflow in a Colab-oriented environment.

### `results/`

Committed experiment outputs used to generate the reported analyses.

* `figures/` — generated plots and visualizations.
* `reference/` — reference measurements used for comparison and validation.
* `runs/` — structured experiment outputs, including serialized JSON/CSV results for the main ImageNet-1k, causal, SAE, replication, and synthetic experiments.

The results directory is organized by experiment family rather than by paper figure; individual result files correspond to the experiment scripts.

### `tests/`

Unit, integration, and end-to-end smoke tests covering the core metrics, SAE implementation, interventions, activation patching, probing, and experiment utilties.

### `paper/`

Full manuscript.

## References

- SAE-related references: BatchTopK (arXiv:2412.06410), TopK / Scaling SAEs (arXiv:2406.04093), JumpReLU
(arXiv:2407.14435), Gated SAEs (arXiv:2404.16014), Matryoshka SAEs
(arXiv:2503.17547), SAEBench (arXiv:2503.09532), Prisma (arXiv:2504.19475),
transcoders (arXiv:2406.11944), end to end SAEs (arXiv:2405.12241),
crosscoders (transformer-circuits.pub, 2024).

- Mannes. Positional Encodings Anchor Spatial Structure in Vision Transformers. arXiv 2606.00124.
- Dosovitskiy et al. An Image is Worth 16x16 Words. ICLR 2021.
- Heo et al. Rotary Position Embedding for Vision Transformer. ECCV 2024.
- Dong, Cordonnier, Loukas. Attention is not all you need: pure attention loses rank doubly exponentially with depth. ICML 2021.
- Hendrycks, Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019.
- Roy, Vetterli. The effective rank: a measure of effective dimensionality. 2007.