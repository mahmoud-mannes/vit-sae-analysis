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

**Mahmoud Mannes**
**Aravind Kannappan**
**Nikhil Maturi**
**Jiwon Jeong**
**Parva Mehta**

## Models and data

| Role | Model | Position encoding | Source |
| --- | --- | --- | --- |
| APE | `google/vit-base-patch16-224` | learned absolute | transformers |
| RoPE | `vit_base_patch16_rope_224.naver_in1k` | rotary | timm |
| APE (replication) | DINOv1 ViT-B/16 | learned absolute | timm |
| RoPE (replication) | DINOv3 ViT-B/16 | rotary | timm |

All ImageNet-1k, streamed from the Hugging Face Hub.

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

## Part 1 — APE: a sparse, causal, permanently destructible code

- SAE feature analysis: explicit row/column features, high
  row/column selectivity.
- Dose-response ablation (linear decodability):

  | Features ablated | Peak top-1 accuracy |
  | --- | --- |
  | 0 (baseline, no reconstruction) | 100% |
  | 0 (SAE reconstruction) | ~100% |
  | 0 (20 random features) | ~100% |
  | 4 | 95.93% |
  | 8 | 60.5% |
  | 12 | 44% |
  | 16 | 33.2% |
  | 20 | 22.9% |

- Component localization (windowed ablation): attention builds the early
  peak (survives all MLP ablation, only moves under attention ablation);
  middle MLPs drive the later decay; late attention only
  softens, doesn't flatten, the decay.
- Rank dissociation: `mlp_zero_all` collapses effective rank 154→43 while
  SSDC-under-RPI stays highest — the two probes measure different things.
- DINOv1 replication: main SAE story replicated.

## Part 2 — RoPE: where the SAE route broke, and why that's informative

**First pass, SAE-based.** positional features identified via
selectivity scoring, visually messier than APE's row/column structure, no
clean geometric form. Ablating them: SSDC drops `0.421 → 0.096` at the
intervention layer, and appeared to recover within a few blocks.

## The reconstruction confound

The apparent RoPE "recovery" was largely an artifact of comparing against the
SAE's own reconstructed baseline rather than the untouched model — the
reconstruction step inflates late-layer SSDC by up to `+0.21`, enough that
the *ablated* model reads as having more structure than a model that was
never touched past a certain layer.

The same confound showed up independently in linear-decodability space: the
SAE-reconstruction baseline alone (zero features ablated) collapsed RoPE
decodability from `38.7%` to `26.4%`, before any causal
claim about specific features. It is important to note this is still significantly
above random chance, as there are `197-201` possibilities (varies between models). This means
the probability of a random chance prediction is roughly equal to `0.5%`

**What this motivated:** moving the causal claims off SAE-based ablation
entirely for RoPE, onto an intervention with zero reconstruction error.

## Part 3 — The rotary-basis intervention

RoPE's positional vocabulary is architecturally exhaustive and directly
addressable: `[TODO: N axes × N frequency bands]` = `[TODO: N]` rotation
planes, shared across all heads and layers on the Axial checkpoint (verified
against the installed timm source — see `[TODO: file]`). Setting a plane's
sine to 0 and cosine to 1 is an exact identity; nothing is reconstructed.

| Workstream | Establishes | Status |
| --- | --- | --- |
| W1 | Correction + reconstruction-artifact diagnostic + normalization protocol | `done` |
| W2 | Necessity — dose-response over axis/frequency/per-head (per-head moved to RoPE-Mixed) | `in progress` |
| W3 | Necessity via persistent vs. single-layer ablation; suffix sweep is the primary measurement | `in progress` |
| W4 | Sufficiency — does restoring the attention output alone restore structure | `in progress` |


## Independent evidence for redundancy, at the attention-block level

- Ablating RoPE's identified positional features directly at the attention
  block output has a real, modest effect on downstream spatial structure
  (attenuated by redundancy); the equivalent APE intervention has almost
  none.
- Layer-count-dependent feature survival: ablating any single layer, or
  layers `[0,1,2]`, leaves `9/11` features intact; only ablating all of
  `[0,1,2,3,4]` collapses this to `2/11`. `[TODO: run the missing
  intermediate conditions (3 and 4 layers) before treating this as a
  clean threshold rather than a two-tier robustness structure. Better to rerun this whole experiment once again]`

## Cross-model replication and the open discrepancy

DINOv1/DINOv3 replication: qualitative pattern holds, but DINOv3's
near-ceiling baseline SSDC (`0.92-0.96`) leaves little room to show a drop,
and the feature-ablation effect size is significantly more modest compared to the original
naver/timm RoPE model, but still real. **Explanation currently under progress**

## Sparse autoencoders

In-depth discussion of SAE implementation and results can be found in the docs directory.

## Repository layout

`[TODO: regenerate this whole block from the actual current tree —
research_notes/, CAUSAL_RESULTS.md, SAE_PLAN.md, activation_patching.py,
position_alignment_patching.py, attention_output_analysis.py,
ablation_sweep.py, position_probe.py all exist in your pasted version but
not in what's currently live on main]`

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