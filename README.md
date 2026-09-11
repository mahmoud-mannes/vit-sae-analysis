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

```
├── CODEOWNERS
├── README.md
├── docs
│   ├── CAUSAL_FOLLOWUPS.md
│   ├── CAUSAL_RESULTS.md
│   ├── DISCUSSION.md
│   ├── EXPERIMENTS.md
│   ├── LINEAR_PROBE.md
│   ├── METHODS.md
│   ├── SAE_PLAN.md
│   └── SAE_RESULTS.md
├── notebooks
│   └── vit_ssdc_ablation_colab.ipynb
├── project_code
│   └── src
│       ├── SAE
│       │   ├── __init__.py
│       │   ├── activation_store.py
│       │   ├── activation_store_scaled.py
│       │   ├── benchmark_synthetic.py
│       │   ├── extract.py
│       │   ├── metrics.py
│       │   ├── run_real.py
│       │   ├── sae.py
│       │   └── train.py
│       ├── SAE_causal
│       │   ├── feature_ablation.py
│       │   └── feature_ablation_hook.py
│       ├── SAE_feature_analysis
│       │   ├── activation_extraction.py
│       │   ├── top_candidates.py
│       │   ├── top_selective_features.py
│       │   └── visualize_feature_activation.py
│       ├── experiments
│       │   ├── __init__.py
│       │   ├── ablation_layerwise.py
│       │   ├── ablation_sweep.py
│       │   ├── activation_patching.py
│       │   ├── attention_output_analysis.py
│       │   ├── common.py
│       │   ├── effective_rank_probe.py
│       │   ├── plot_causal_results.py
│       │   ├── position_alignment_patching.py
│       │   ├── reproduce_robustness.py
│       │   └── reproduce_ssdc.py
│       ├── interventions
│       │   ├── __init__.py
│       │   ├── ablation.py
│       │   └── corruptions.py
│       ├── main
│       │   ├── load_models.py
│       │   ├── make_imagenet_c.py
│       │   ├── model.py
│       │   └── prep_data.py
│       └── metrics
│           ├── effective_rank.py
│           ├── position_probe.py
│           ├── robustness.py
│           └── ssdc.py
├── requirements.txt
├── results
│   ├── figures
│   │   ├── APE_MLP_ablation_dinov1.png
│   │   ├── APE_attention_ablation_dinov1.png
│   │   ├── ROPE_MLP_ablation_dinov3.png
│   │   ├── RoPE_attention_ablation_dinov3.png
│   │   ├── ablation_attn_zero_rpi.png
│   │   ├── ablation_mlp_keep_rpi.png
│   │   ├── ablation_mlp_zero_rpi.png
│   │   ├── causal_attention_output_ssdc.png
│   │   ├── causal_attention_position_probes.png
│   │   ├── causal_final_layer_ablation.png
│   │   ├── causal_final_layer_baselines.png
│   │   ├── causal_final_layer_patching.png
│   │   ├── causal_peak_layer_ablation.png
│   │   ├── causal_peak_layer_patching.png
│   │   ├── causal_position_alignment.png
│   │   ├── effective_rank_ape.png
│   │   ├── robustness_fragility.png
│   │   ├── sae_real_frontier.png
│   │   ├── sae_synthetic_frontier.png
│   │   ├── ssdc_ape.png
│   │   └── ssdc_rope.png
│   ├── reference
│   │   ├── robustness_reference.json
│   │   └── ssdc_reference.json
│   └── runs
│       ├── SAE_20k_images_analysis
│       │   ├── original_results
│       │   │   ├── feature_ablation_residual.json
│       │   │   ├── mean_row_column_selectivity_attention.json
│       │   │   ├── mean_row_column_selectivity_residual.json
│       │   │   ├── top_selective_features_per_position_attention.json
│       │   │   └── top_selective_features_per_position_residual.json
│       │   └── second_seed
│       │       ├── feature_ablation_second_seed_residual.json
│       │       ├── mean_row_column_selectivity_attention_second_seed.json
│       │       ├── mean_row_column_selectivity_residual_second_seed.json
│       │       ├── top_selective_features_per_position_attention_second_seed.json
│       │       └── top_selective_features_per_position_residual_second_seed.json
│       ├── imagenet1k_test
│       │   ├── sae_scaled_runs_20k_images.json
│       │   ├── sae_scaled_runs_20k_images_attention.json
│       │   ├── sae_scaled_runs_20k_images_attention_second_seed.json
│       │   └── sae_scaled_runs_20k_images_second_seed,json
│       ├── imagenet1k_val
│       │   ├── ablation_ape.json
│       │   ├── ablation_layerwise_ape_second_seed.json
│       │   ├── ablation_layerwise_rope_second_seed.json
│       │   ├── causal_followups
│       │   │   ├── attention_output_and_probes_seed0.json
│       │   │   ├── attention_output_and_probes_seed1.json
│       │   │   ├── final_layer_activation_patching_ape_seed0.json
│       │   │   ├── final_layer_activation_patching_ape_seed1.json
│       │   │   ├── final_layer_activation_patching_rope_seed0.json
│       │   │   ├── final_layer_activation_patching_rope_seed1.json
│       │   │   ├── final_layer_zero_ablation_ape_seed0.json
│       │   │   ├── final_layer_zero_ablation_ape_seed1.json
│       │   │   ├── final_layer_zero_ablation_rope_seed0.json
│       │   │   ├── final_layer_zero_ablation_rope_seed1.json
│       │   │   ├── manifest.json
│       │   │   ├── peak_layer_activation_patching_ape_seed0.json
│       │   │   ├── peak_layer_activation_patching_ape_seed1.json
│       │   │   ├── peak_layer_activation_patching_rope_seed0.json
│       │   │   ├── peak_layer_activation_patching_rope_seed1.json
│       │   │   ├── peak_layer_zero_ablation_ape_seed0.json
│       │   │   ├── peak_layer_zero_ablation_ape_seed1.json
│       │   │   ├── peak_layer_zero_ablation_rope_seed0.json
│       │   │   ├── peak_layer_zero_ablation_rope_seed1.json
│       │   │   ├── position_alignment_ape_seed0.json
│       │   │   ├── position_alignment_ape_seed1.json
│       │   │   ├── position_alignment_rope_seed0.json
│       │   │   └── position_alignment_rope_seed1.json
│       │   ├── effective_rank_ape.json
│       │   ├── robustness.json
│       │   ├── sae_real_benchmark.json
│       │   └── ssdc.json
│       └── synthetic
│           └── sae_benchmark.json
└── tests
    ├── e2e_smoke.py
    ├── fake_vit.py
    ├── test_ablation_readouts.py
    ├── test_activation_patching.py
    ├── test_attention_output_analysis.py
    ├── test_causal_results.py
    ├── test_coordinate_probes.py
    ├── test_core.py
    ├── test_experiment_common.py
    ├── test_position_alignment_patching.py
    ├── test_sae.py
    └── test_ssdc_accumulator.py
```

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