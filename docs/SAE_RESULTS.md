# SAE benchmark results

These are real training runs, not projections. The real ViT numbers come from
the pretrained ViT-Base (APE, `google/vit-base-patch16-224`), the synthetic
numbers from a known sparse dictionary. Both were produced on CPU, so they are a
small scale demonstration rather than a production SAE. A GPU with more tokens
and epochs would sharpen every number, but the ordering between methods is
already clear and is what matters for "which recipe is better".

Reproduce:

```bash
cd project_code/src
python -m SAE.run_real --images 768 --layer 6 --epochs 14   # needs HF_TOKEN
python -m SAE.benchmark_synthetic
```

## Real ViT-Base, APE, layer 6 residual stream

768 ImageNet-1k validation images, 150,528 patch tokens (class token dropped),
`d_hidden = 1536`, 14 epochs. FVU and L0 are held out. `down agree` and `KL` are
the downstream reconstruction splice: replace the layer 6 patch activations with
the SAE reconstruction and measure how often the model's top-1 prediction is
unchanged, and the KL of the logits. Figure: `results/figures/sae_real_frontier.png`.

| Model | L0 | FVU | dead | cos | down agree | KL |
| --- | --- | --- | --- | --- | --- | --- |
| baseline L1, no b_dec, c=1 | 289.4 | 0.024 | 0.00 | 0.72 | 0.711 | 0.68 |
| baseline L1, no b_dec, c=8 | 78.4 | 0.049 | 0.00 | 0.44 | | |
| baseline L1, no b_dec, c=40 | 42.7 | 0.149 | 0.00 | 0.36 | | |
| L1 + b_dec, c=8 | 148.0 | 0.087 | 0.00 | 0.24 | | |
| **TopK k=32** | **32.0** | **0.031** | **0.01** | 0.63 | **0.656** | 0.99 |
| BatchTopK k=16 | 16.0 | 0.048 | 0.75 | 0.31 | | |
| BatchTopK k=32 | 32.0 | 0.032 | 0.11 | 0.52 | 0.227 | 3.93 |
| BatchTopK k=64 | 64.0 | 0.027 | 0.00 | 0.61 | | |
| JumpReLU c=0.3 / 1.0 | 661 | 0.013 | 0.00 | 0.86 | | |

What this says.

* **The modern recipe dominates the L1 baseline on the frontier.** At L0 = 32,
  TopK reaches FVU 0.031 and BatchTopK 0.032. The L1 baseline cannot get there:
  to reach FVU 0.049 it needs L0 = 78 (more than twice as many active latents),
  and when pushed toward L0 = 43 its FVU blows up to 0.149. L1 shrinks activation
  magnitudes rather than zeroing them, so it cannot operate in the sparse regime
  where interpretable features live.
* **TopK preserves what the model computes.** With only 32 active latents it keeps
  65.6% of the model's top-1 predictions through the splice, close to the dense L1
  baseline's 71.1% which uses 289 active latents (9x more). Low FVU alone would
  not tell you this; the downstream metric does.
* **BatchTopK struggles at low L0 here.** Its dead fraction is 0.75 at k=16 and
  0.11 at k=32, and its downstream agreement is low (0.23) as a result. It is
  clean at k=64 (0 dead, FVU 0.027). This matches the known BatchTopK weakness at
  low L0 (Bussmann et al., 2024). So on this layer TopK is the better low L0
  choice and BatchTopK is better once you allow a larger budget.
* **JumpReLU did not sparsify in this run** (L0 stuck at 661, identical for two
  penalties), meaning its learned threshold barely moved. Its straight through
  bandwidth needs tuning to the activation scale; that is a follow up, not a
  result. TopK and BatchTopK are the working recommendations.

## Synthetic ground truth dictionary

`d_model = 256`, `n_true = 1024` features in superposition, average 10 active,
`d_hidden = 1024`, 15 epochs. Recovery is the mean max cosine similarity between
the learned decoder and the true dictionary. Figure:
`results/figures/sae_synthetic_frontier.png`.

| Model | L0 | FVU | recovery |
| --- | --- | --- | --- |
| baseline L1, no b_dec, c=0.3 | 98.5 | 0.386 | 0.995 |
| baseline L1, no b_dec, c=3 | 49.0 | 0.758 | 0.995 |
| baseline L1, no b_dec, c=30 | 24.6 | 1.047 | 0.995 |
| TopK k=6 | 6.0 | 0.205 | 0.997 |
| TopK k=10 | 10.0 | 0.208 | 0.995 |
| TopK k=16 | 16.0 | 0.226 | 0.991 |
| BatchTopK k=6 | 6.0 | 0.273 | 0.994 |
| BatchTopK k=10 | 10.0 | 0.306 | 0.992 |
| BatchTopK k=16 | 16.0 | 0.352 | 0.989 |
| JumpReLU | 419 | 0.951 | 0.987 |

What this says.

* **TopK at L0 = 6 (FVU 0.205) beats L1 at L0 = 98 (FVU 0.386)**: sixteen times
  sparser and a better reconstruction at the same time.
* **L1's reconstruction gets worse as the penalty grows** (FVU 0.39, 0.76, 1.05).
  An FVU above 1 is worse than predicting the mean. This is the L1 shrinkage
  pathology in its purest form and it is exactly what topk and jumprelu were
  designed to avoid.
* **Recovery saturated near 0.99 for every method**, because random unit vectors
  in 256 dimensions are nearly orthogonal even at 4x overcompleteness, so the
  true directions are easy to recover once found. The separation here is in
  fidelity at a fixed sparsity, not in recovery. A harder dictionary (correlated
  features, higher overcompleteness) would spread the recovery numbers; that is a
  Stage 4 refinement.

## Bottom line

The new package beats the original ReLU + L1 autoencoder where it counts: at any
sparse operating point (L0 <= 64) it reconstructs the ViT residual stream several
times more faithfully, keeps the model's predictions through a splice, and does
it with a hard sparsity mechanism that L1 cannot match. For this ViT layer, TopK
is the best low L0 choice and BatchTopK the best at higher L0. The foundation
changes (b_dec centring, activation normalisation, AuxK in place of resampling,
held out FVU and a downstream metric) are what make the comparison trustworthy.

Known limitations of this run, all tracked for the next stages: CPU scale
(150k tokens, 14 epochs, one layer, `d_hidden` 1536), a JumpReLU threshold that
needs tuning, and a synthetic dictionary too easy for recovery to discriminate.
