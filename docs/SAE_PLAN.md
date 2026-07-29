# Training a better SAE, and where to take it next

This document is the plan behind the `project_code/src/SAE` package. It records
what was wrong with the original autoencoder, what the new package changes and
why, how the change is evaluated, and the research directions that follow.

## What the original SAE was

The original `train_SAE.py` is a single hidden layer ReLU dictionary with an L1
penalty, a unit norm decoder, and Anthropic style dead feature resampling. It is
a faithful 2023 recipe. Concrete gaps:

1. **No pre-encoder bias.** The encoder never saw a centred input, so the large
   shared mean of the residual stream leaked into every feature.
2. **L1 sparsity.** L1 causes activation shrinkage, which flattens the
   sparsity/fidelity frontier and makes a given L0 cost more reconstruction than
   it should. It is also hard to tune: the coefficient that yields a target L0
   depends on the activation scale (our real run shows L1 at one coefficient
   sitting at L0 ~ 675 while topk hits an exact L0 of 32).
3. **Single batch `EV` proxy.** Reconstruction was reported on the training
   batch with a scalar variance denominator, not held out fraction of variance
   unexplained, and there was no downstream faithfulness check.
4. **One pass, no normalisation, no held out split** in the data pipeline.

## What the new package changes

`project_code/src/SAE/` keeps one `SAE` class whose sparsity mechanism is a
constructor argument, so every variant trains and evaluates through the same
code and the comparison is apples to apples.

* **`b_dec` centring and tied init** (`sae.py`). The encoder sees `x - b_dec`,
  the decoder adds it back, `b_dec` starts at the data mean, and `W_enc` starts
  as `W_dec.T`.
* **Modern sparsity** (`sae.py`): `relu_l1` (the old baseline), `topk` and
  `batchtopk` (set L0 directly, no penalty tuning, no shrinkage), and `jumprelu`
  (a learned per feature threshold via a straight through estimator). On the
  measured runs (see [SAE_RESULTS.md](SAE_RESULTS.md)) TopK and BatchTopK both
  dominate L1; TopK was cleanest at low L0 (near zero dead latents) and BatchTopK
  best at a larger budget. BatchTopK converts to a single global threshold at
  inference and runs as a per token JumpReLU.
* **AuxK auxiliary loss** (`sae.py`, `train.py`) replaces resampling: dead
  latents are asked to reconstruct the residual, which revives them without hand
  tuned resets.
* **Activation store** (`activation_store.py`): scalar normalisation so
  `E[||x||] = sqrt(d_model)`, a held out split, and shuffled multi epoch
  iteration.
* **Honest metrics** (`metrics.py`): held out FVU, L0, dead fraction, cosine,
  ground truth recovery for the synthetic benchmark, and a downstream
  reconstruction splice (top-1 agreement and logit KL) for the real ViT.

## How "better" is proven

* **Synthetic ground truth benchmark** (`benchmark_synthetic.py`). Data is built
  from a known sparse dictionary in a superposition regime, so recovery of the
  true features is measurable (mean max cosine similarity), which no real
  activation set can give. Run: `python -m SAE.benchmark_synthetic`.
* **Real ViT benchmark** (`run_real.py`). Streams ImageNet, captures the residual
  stream entering a chosen block of the pretrained ViT-Base, trains every variant
  and reports the held out frontier plus the downstream splice. Run:
  `python -m SAE.run_real --images 512 --layer 6 --epochs 10`.

Results are written to `results/runs/` and `results/figures/`, and summarised in
[SAE_RESULTS.md](SAE_RESULTS.md).

## Stages

* **Stage 0-2 (this package).** Foundation (store, metrics), architecture
  (`b_dec`, BatchTopK/JumpReLU, AuxK), and the two benchmarks.
* **Stage 3 (vision and position specific).** Vision SAEs run at higher L0 than
  language SAEs (Joseph et al., Prisma, arXiv:2504.19475), so sweep L0 higher.
  Train at the layers this project cares about (the early attention peak and the
  middle MLP decay). Use RPI vs non RPI activations to label latents as index
  anchored or content driven, and turn the existing row/column position
  selectivity into a logged model selection metric.
* **Stage 4 (rigorous eval).** A SAEBench style harness (arXiv:2503.09532):
  FVU/L0 Pareto, downstream ImageNet delta accuracy and KL, feature absorption,
  and an automated interpretability score over top activating image patches.

## Research directions (from current literature)

1. **Crosscoder model diffing of APE vs RoPE.** Train one acausal crosscoder over
   both models' residual streams to separate shared from model specific position
   features. This is the most on thesis idea: it turns the aggregate SSDC/RPI
   comparison into a feature level account. (Anthropic crosscoders, 2024;
   cross architecture crosscoders, arXiv:2602.11729.)
2. **Transcoders / skip transcoders on the middle MLPs.** The ablation study found
   the middle MLPs drive the SSDC decay. A transcoder that approximates those
   MLPs reads the features doing it, turning a correlational ablation into a
   mechanism. (arXiv:2406.11944, arXiv:2501.18823.)
3. **Matryoshka BatchTopK** if position features split or absorb at large
   dictionary size: nested dictionaries give coarse (row/band/quadrant) to fine
   (exact patch) position features with less absorption. (arXiv:2503.17547.)
4. **RPI contrastive positional dictionary.** Score every latent on RPI vs non RPI
   activation to define a per layer positional fraction, and overlay it on the
   SSDC under RPI curve.
5. **End to end SAEs.** Train to preserve the ImageNet prediction (logit KL), not
   only reconstruction, for features faithful to the computation. (arXiv:2405.12241.)

## References

BatchTopK (arXiv:2412.06410), TopK / Scaling SAEs (arXiv:2406.04093), JumpReLU
(arXiv:2407.14435), Gated SAEs (arXiv:2404.16014), Matryoshka SAEs
(arXiv:2503.17547), SAEBench (arXiv:2503.09532), Prisma (arXiv:2504.19475),
transcoders (arXiv:2406.11944), end to end SAEs (arXiv:2405.12241),
crosscoders (transformer-circuits.pub, 2024).
