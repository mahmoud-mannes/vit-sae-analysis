# Question: How does ablating positional features from the residual stream affect linear decodability of token positions?

We extract activations from APE- and RoPE-based models under the RPI intervention, then train a linear probe to predict the exact position of each token. We use peak top-1 accuracy as a secondary, confirmatory metric for SSDC, since it directly measures how linearly decodable positional information is from the residual stream. As with SSDC, we use RPI to eliminate content confounds and isolate positional structure.

Throughout this document, **"seed" refers only to an internal repository label used to distinguish model configurations. It does not refer to a random seed, and the terminology will not be used in the paper.** The mapping is:

| Internal seed label | Model                              |
| ------------------- | ---------------------------------- |
| Seed 1              | `google/vit-base-patch16-224`      |
| Seed 2              | `vit_base_patch16_224.dino`        |
| Seed 3              | `vit_base_patch16_224.augreg_in1k` |
| Seed 4              | `vit_base_patch16_224.sam_in1k`    |

## Hyperparameters

* `number_images = 1000` (except APE first seed: 10,000)
* `lr = 5e-3`
* `num_passes = 20`
* `batch_size = 512`
* `RPI = True`

## APE Layer 2 — Seed 1

| Intervention                       | Peak linear probe accuracy |
| ---------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction)   |                     99.99% |
| Baseline (SAE reconstruction)      |                      99.7% |
| 20 random features ablated         |                      99.1% |
| 4 top positional features ablated  |                     95.93% |
| 8 top positional features ablated  |                      60.5% |
| 12 top positional features ablated |                      44.0% |
| 16 top positional features ablated |                      33.2% |
| 20 top positional features ablated |                      22.9% |

## RoPE Layer 4 — Seed 1

| Intervention                     | Peak linear probe accuracy |
| -------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction) |                      38.7% |
| Baseline (SAE reconstruction)    |                      26.4% |

## RoPE Layer 4 — Seed 2

| Intervention                     | Peak linear probe accuracy |
| -------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction) |                      35.3% |
| Baseline (SAE reconstruction)    |                      27.5% |

## APE Layer 2 — Seed 2

| Intervention                       | Peak linear probe accuracy |
| ---------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction)   |                      95.9% |
| Baseline (SAE reconstruction)      |                      82.1% |
| 20 random features ablated         |                      83.4% |
| 4 top positional features ablated  |                     58.69% |
| 8 top positional features ablated  |                      28.4% |
| 12 top positional features ablated |                      18.2% |
| 16 top positional features ablated |                       6.6% |
| 20 top positional features ablated |                       4.3% |

## APE Layer 2 — Seed 3

| Intervention                       | Peak linear probe accuracy |
| ---------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction)   |                     99.94% |
| Baseline (SAE reconstruction)      |                     99.96% |
| 20 random features ablated         |                      99.8% |
| 4 top positional features ablated  |                      95.0% |
| 8 top positional features ablated  |                      64.7% |
| 12 top positional features ablated |                      53.7% |
| 16 top positional features ablated |                      36.8% |
| 20 top positional features ablated |                     29.25% |

## APE Layer 2 — Seed 4

| Intervention                       | Peak linear probe accuracy |
| ---------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction)   |                       100% |
| Baseline (SAE reconstruction)      |                       100% |
| 33 random features ablated         |                       100% |
| 8 top positional features ablated  |                      85.6% |
| 16 top positional features ablated |                      32.5% |
| 24 top positional features ablated |                     14.68% |
| 33 top positional features ablated |                      0.05% |


## Interpretation

The APE results generalize substantially across model configurations. All four APE models exhibit very high baseline linear positional decodability, while ablating the identified positional features produces a strong and systematic reduction in probe accuracy. In contrast, ablating a comparable number of randomly selected features has little to no effect: the random-feature controls remain close to their corresponding SAE-reconstructed baselines.

This pattern is observed across substantially different APE model configurations, including the standard pretrained ViT, DINO, AugReg, and SAM variants. The additional models therefore provide a cross-model generalization check that the observed effect is not specific to a single pretrained model.

The magnitude of the effect varies across models. For example, Seed 3 falls from 99.96% after SAE reconstruction to 29.25% after ablating 20 positional features, while Seed 4 falls from 100% to 0.05% after ablating 33 positional features. This variation is expected given differences in the number and relative importance of localized positional features across models; the key qualitative result is the consistent separation between targeted positional-feature ablations and random-feature controls.

The Seed 4 result is particularly strong as a specificity control: ablating 33 identified positional features reduces peak probe accuracy from 100% to 0.05%, whereas ablating 33 random features leaves accuracy at 100%. Thus, the loss of positional decodability is not explained simply by removing a comparable number of SAE features.

For the original APE model, the same qualitative trend was already present: ablating progressively more identified positional features produced a monotonic decrease in linear positional decodability, while random-feature ablation had comparatively little effect. The additional model configurations reproduce this relationship.

Despite the broadly similar SAE performance metrics (EV, L0, dead-feature fraction, etc.) at the respective layers, SAE reconstruction has a substantially larger effect on linear positional decodability in the RoPE models.

There are two non-competing explanations:

1. **The effective reconstruction quality may be worse for RoPE in a way that our standard reconstruction metrics do not capture.**
2. **RoPE's positional information may be substantially more sensitive to the perturbations in activation statistics introduced by SAE reconstruction.**

Importantly, this should not necessarily be interpreted as general “fragility” or lack of robustness to distributional shifts; it is specifically sensitivity to this particular reconstruction-induced perturbation.

It is also important that RoPE is not completely losing positional information after reconstruction: ~27% top-1 accuracy is still ~54× the random baseline (1/197 ≈ 0.51%). So linear positional decodability is substantially degraded, but not eliminated.

For now, we do not think it is worth expanding the RoPE feature-ablation analysis further. This single-layer result is less directly relevant to the main RoPE mechanistic story, particularly since SAE reconstruction introduces a potential confound. We will likely revisit linear probes in the multi-layer study, using interventions at the rotary-module level that bypass SAE reconstruction error.

The additional APE experiments are best treated as a **cross-model generalization test** near the end of the analysis. They strengthen the claim that the targeted-feature ablation effect is reproducible across different pretrained APE ViT configurations.