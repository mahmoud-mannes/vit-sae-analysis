# Question: How does ablating positional features from the residual stream affect linear decodability of token positions?

We extract activations from APE- and RoPE-based models under the RPI intervention, then train a linear probe to predict the exact position of each token. We use peak top-1 accuracy as a secondary, confirmatory metric for SSDC, since it directly measures how linearly decodable positional information is from the residual stream. As with SSDC, we use RPI to eliminate content confounds and isolate positional structure.

**Hyperparameters**

* `number_images = 1000` (except APE first seed: 10,000)
* `lr = 5e-3`
* `num_passes = 20`
* `batch_size = 512`
* `RPI = True`

### APE Layer 2 — Seed 1

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

### RoPE Layer 4 — Seed 1

| Intervention                     | Peak linear probe accuracy |
| -------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction) |                      38.7% |
| Baseline (SAE reconstruction)    |                      26.4% |

### RoPE Layer 4 — Seed 2

| Intervention                     | Peak linear probe accuracy |
| -------------------------------- | -------------------------: |
| Baseline (no SAE reconstruction) |                      35.3% |
| Baseline (SAE reconstruction)    |                      27.5% |

### APE Layer 2 — Seed 2

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

## Interpretation

Despite the broadly similar SAE performance metrics (EV, L0, dead-feature fraction, etc.) at the respective layers, SAE reconstruction has a substantially larger effect on linear positional decodability in the RoPE models.

There are two non-competing explanations:

1. **The effective reconstruction quality may be worse for RoPE in a way that our standard reconstruction metrics do not capture.**
2. **RoPE's positional information may be substantially more sensitive to the perturbations in activation statistics introduced by SAE reconstruction.** Importantly, this should not necessarily be interpreted as general “fragility” or lack of robustness to distributional shifts; it is specifically sensitivity to this particular reconstruction-induced perturbation.

It is also important that RoPE is not completely losing positional information after reconstruction: ~27% top-1 accuracy is still ~54× the random baseline (1/197 ≈ 0.51%). So linear positional decodability is substantially degraded, but not eliminated.

For now, I don't think it is worth expanding the RoPE feature-ablation analysis further. This single-layer result is less directly relevant to the main RoPE mechanistic story, particularly since SAE reconstruction introduces a potential confound. We will likely revisit linear probes in the multi-layer study, using interventions at the rotary-module level that bypass SAE reconstruction error.

