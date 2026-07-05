# Experiments

This document lays out the layer windowed ablation experiment, the questions it
answers, the signatures that distinguish the competing explanations, and the
directions this can grow into. The predictions below are hypotheses to test with
the notebook. They are not measured results.

## Background from the reference run

For the APE model the SSDC under RPI curve rises to an early peak around blocks 2
to 3 and then decays over the later blocks, from about 0.66 down to about 0.13.
For the RoPE model the same curve accumulates more gradually and peaks later,
around block 5, then decays more mildly. Two facts stand out for APE:

- The index anchored structure is built fast, in the early blocks.
- It then decays across the later blocks.

The starting claims are that MLPs carry the recovery and that attention drives
the later decay. The whole model versions of those claims are coarse. This
experiment localizes them.

## The two explanations for the early peak

APE peaks early. There are two clean explanations.

1. **Early layers are special.** Something about the early MLP blocks
   specifically builds the index anchored structure. Move the computation later
   and it will not build the same way.
2. **First blocks encountered.** The peak forms at whatever MLP blocks the tokens
   meet first. Early is incidental. The first surviving MLPs, wherever they sit,
   would build it.

These make different predictions under ablation, so they are separable.

## Conditions

Windows: early `[0,1,2,3]`, mid `[4,5,6,7]`, late `[8,9,10,11]`. All conditions
are measured through SSDC under RPI, since that is where the early peak and the
later decay both live for APE. The baseline clean SSDC is recorded for context.

| Name | Component | Layers | Mode |
| --- | --- | --- | --- |
| `baseline` | none | | |
| `mlp_zero_early` | MLP | early | zero |
| `mlp_zero_mid` | MLP | mid | zero |
| `mlp_zero_late` | MLP | late | zero |
| `mlp_zero_all` | MLP | all | zero |
| `mlp_keep_early` | MLP | early | keep only |
| `mlp_keep_mid` | MLP | mid | keep only |
| `mlp_keep_late` | MLP | late | keep only |
| `attn_zero_early` | attention | early | zero |
| `attn_zero_mid` | attention | mid | zero |
| `attn_zero_late` | attention | late | zero |
| `attn_zero_all` | attention | all | zero |

## Question 1: is the early peak unique to the early MLPs

Compare `mlp_zero_early` with `mlp_zero_late`.

- If `mlp_zero_early` removes or badly delays the early peak while `mlp_zero_late`
  leaves it in place, the early peak is specifically an early MLP phenomenon.
- If both leave the early peak roughly intact, no single window owns it and the
  recovery is distributed.

`mlp_zero_all` is the control. It should crush SSDC under RPI toward zero at all
depths if MLPs are the carrier of the recovery.

## Question 2: early layers, or just the first MLPs encountered

This is the `keep_only` probe. Keep MLPs alive in exactly one window and ablate
the rest.

- If the peak forms in the early blocks under `mlp_keep_early`, in the mid blocks
  under `mlp_keep_mid`, and in the late blocks under `mlp_keep_late`, then the
  peak follows the first surviving MLPs. That supports *first blocks
  encountered*. Position in the stack does not matter, only being the first MLPs
  the tokens meet.
- If a peak only appears when the early window is kept, and `mlp_keep_mid` or
  `mlp_keep_late` produce a weak or absent peak, then the early layers are doing
  something the later layers cannot. That supports *early layers are special*.

Watching `peak_layer` in the printed summary across the three keep only
conditions is the direct test. A `peak_layer` that tracks the kept window is the
first blocks encountered signature.

## Question 3: does removing attention only in later layers still destroy the decay

Compare `attn_zero_all`, `attn_zero_late`, and `attn_zero_early`.

- If `attn_zero_all` flattens the decay, so SSDC under RPI stays high through the
  late blocks instead of falling, attention is what erodes the index anchored
  structure late.
- If `attn_zero_late` alone also flattens the decay, the decay is a late block,
  attention driven effect. Late attention specifically is what mixes tokens and
  washes out the position anchored similarity.
- If `attn_zero_late` does not flatten the decay but `attn_zero_early` changes the
  peak, attention's role is front loaded and the decay has another cause.

The `decay` column in the summary (peak minus final) is the number to watch. A
baseline with large decay, an `attn_zero_all` with small decay, and an
`attn_zero_late` that also shrinks decay together pin the decay on late attention.

## Reading the summary table

`ablation_layerwise.py` prints one row per condition with these fields.

- `peak` and `peak_layer`: height and depth of the SSDC under RPI maximum.
- `delta`: SSDC at block 1 minus block 0, the immediate recovery after the first
  block.
- `decay`: peak minus final, the size of the later layer fall.
- `final`: SSDC at the last block.
- `auc`: mean SSDC over depth, a single number for total recovery.

## Extension: rank collapse as the mechanism

Dong et al. (2021) prove that stacked attention without MLPs and skip connections
drives token representations toward rank one with depth. That is a reason to
expect MLP ablation to hurt any structure carried by the residual stream. The
prediction is concrete. Under `mlp_zero_all` the effective rank should fall with
depth, and it should fall in the same blocks where SSDC under RPI collapses. If
those two curves line up, the loss of index anchored structure is explained by a
loss of representational capacity, not just correlated with it.
`effective_rank_probe.py` runs baseline, `mlp_zero_all`, and `attn_zero_all` for
this comparison.

## Extension: RoPE cross check

Repeat the ablation study on the RoPE model. RoPE injects position inside
attention, so its recovery accumulates with depth rather than peaking early. Two
predictions follow. First, attention ablation should hurt RoPE more directly than
APE, because RoPE's positional signal lives in attention. Second, the first
blocks encountered probe should behave differently, since there is no early
additive injection to anchor an early peak. Running `ablation_layerwise.py
--model rope` produces the curves for this comparison.

## Extension: input level content disruption

`interventions/corruptions.py` adds a grid patch shuffle. It scrambles where local
content sits while keeping the global palette, so it is an input level analogue of
RPI. Measuring fragility under patch shuffle next to Gaussian blur separates
reliance on local content from sensitivity to low frequency degradation.

## Future work

- Sweep the ablation window boundary block by block to find the exact depth where
  the decay switches on, rather than using fixed quartile windows.
- Mean ablate instead of zero ablate, to remove a component's variation while
  keeping its average contribution, and compare the two ablation styles.
- Connect ablation to the sparse autoencoder features in `project_code/src/SAE`.
  Train an SAE on the residual stream at the peak block, then ask which learned
  features are index anchored by measuring their survival under RPI, and whether
  MLP ablation removes exactly those features.
- Extend the fragility panel to more ImageNet-C shifts and relate fragility to the
  per condition SSDC recovery, closing the loop between spatial structure and
  robustness on pretrained models.
