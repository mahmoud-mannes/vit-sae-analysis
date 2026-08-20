# Causal position follow-up experiments

This document covers four follow-up experiments that localize and test spatial
position information in the pretrained APE and RoPE ViTs.

## Shared infrastructure

The experiments use the repository's existing model, intervention, metric, and
data boundaries:

- `main.load_models` loads both ViTs and resolves blocks, attention, MLPs, patch
  embeddings, and prefix-token counts.
- `interventions.ablation.AblationController` performs reversible attention and
  MLP zero ablations.
- `metrics.ssdc.SSDCAccumulator` is the shared implementation for cosine SSDC
  readouts.
- `metrics.position_probe` contains both existing token-position classifiers
  and the held-out row/column ridge probes used by the attention-output study.
- `experiments.common` handles ImageNet loading, reusable batches, deterministic
  RPI permutations, model forwarding, accuracy counts, and JSON output.
- Distribution-shift fragility remains defined by `metrics.robustness`; the
  follow-up runners record classification accuracy and do not reimplement that
  metric.

The existing `LinearProbe` predicts one discrete token-position class. The new
coordinate probe is also linear, but predicts normalized row and column values
with ridge regression. These answer different questions and therefore remain
separate public APIs in `metrics.position_probe`.

## Experiments

### Layer-wise activation patching

`experiments/activation_patching.py` caches a clean component activation and
patches it into the same image under a fixed Random Permutation at Inference
(RPI). It supports residual, attention, MLP, head, and query/key/value targets.

```bash
cd project_code/src
python experiments/activation_patching.py \
  --model ape --components residual attn mlp \
  --number-images 512 --batch-size 32 --shuffle
```

### Independent layer-wise ablation

`experiments/ablation_sweep.py` zeroes one attention or MLP update at a time and
measures block-output SSDC plus top-1 accuracy under normal order and RPI.

```bash
python experiments/ablation_sweep.py \
  --model rope --layers 0 1 2 3 4 5 \
  --ssdc-readout-layers 4 5 \
  --number-images 512 --batch-size 32 --shuffle
```

### Position-alignment controls

`experiments/position_alignment_patching.py` compares an aligned clean donor
with a token-misaligned donor, a donor from another image, and a batch-mean
donor. This separates token alignment from image content.

```bash
python experiments/position_alignment_patching.py \
  --model ape --readout-layers 4 5 \
  --number-images 512 --batch-size 32 --shuffle
```

### Attention-output SSDC and coordinate probes

`experiments/attention_output_analysis.py` measures SSDC at explicit block
stages and fits held-out linear ridge probes for patch row and column. Probe
splits are made by image, probe evaluation uses a different RPI permutation,
and shuffled coordinate targets provide a negative control.

```bash
python experiments/attention_output_analysis.py \
  --model both --number-images 256 --batch-size 32 --half
```

## Reproducibility rules

- Use the official `ILSVRC/imagenet-1k` validation split and a Hugging Face read
  token after accepting the dataset terms.
- Keep image-sample and RPI seeds fixed within a comparison.
- Use a new sample and RPI seed for each repeat.
- Report normal-order accuracy separately from RPI accuracy.
- Treat activation patching as a sufficiency test and zero ablation as a
  necessity test.
- Do not interpret linear-probe decodability alone as causal evidence.

Each runner writes one JSON file under `results/`. The two-seed ImageNet runs
used in the report are curated under
`results/runs/imagenet1k_val/causal_followups/`; figures are in
`results/figures/`, and the results are summarized in
[`docs/CAUSAL_RESULTS.md`](CAUSAL_RESULTS.md).

Regenerate all eight curated figures from those committed JSON runs with:

```bash
cd project_code/src
python experiments/plot_causal_results.py
```

## Tests

Return to the repository root before running the CPU-only tests:

```bash
cd ../..
python tests/test_core.py
```

The focused follow-up suite is:

```bash
for test in \
  tests/test_causal_results.py \
  tests/test_experiment_common.py \
  tests/test_ssdc_accumulator.py \
  tests/test_coordinate_probes.py \
  tests/test_activation_patching.py \
  tests/test_ablation_readouts.py \
  tests/test_attention_output_analysis.py \
  tests/test_position_alignment_patching.py; do
  python "$test"
done
```
