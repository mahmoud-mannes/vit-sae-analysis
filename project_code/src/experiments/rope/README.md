# RoPE interventions

Scripts for the rotary-embedding experiments, W1-A to W7 in the order they were run. The main
script of each stage (`w1a_sae_reanalysis.py`, `w1b_floor_measurement.py`, and the first script of
W2 to W6 in the table) is a notebook exported to a plain script, with `# ---- cell N ----` lines
marking the original cells; the others were written as scripts. The outputs of the runs used in the
paper are in `results/runs/imagenet1k_val/rope_interventions/`.

Run everything from one working directory, stage by stage. Each stage writes to `./<stage>_output/`,
and the helper scripts of a stage read and extend that directory, so `w1b_dinov3_rerun.py` expects
`w1a_output/` and `w1b_output/`, the E1.2 scripts write into `w1b_output/`, and the verify scripts
read `w4_output/` to `w6_output/`. `w2_perblock_gate.py` writes to `./perblock_out/` (or `$OUT_DIR`).

Setup: one H100, timm 1.0.29, PyTorch 2.10. W1-B and E1.2 ran in fp16 through the repo loaders'
`half=True`; W2 onwards ran in float32, and W2 checks that the two agree to within 0.0001 on all four
models. W1-A reads JSON only and needs no GPU. SSDC and probes use 1,000 ImageNet-1k validation
images, the W5 attention statistics 64, and the W6 accuracy runs 5,000. From W2 on, every RoPE model
is loaded with `load_rope(input_size=224)`, including DINOv3 and reg1-gap, whose native input is 256;
W5 and W6 also evaluate at 320. In W1-B and the first E1.2 runs DINOv3 ran on its native 256 grid,
and `w1b_supplementary.py` and `e12_dinov3_224.py` repeat it at 224. W7 ran later on two H100s with
PyTorch 2.14 (same timm), in float32 at 224; its outputs are in `rope_interventions/w7/`.

| Script | What it measures |
| --- | --- |
| `w1a_sae_reanalysis.py` | re-analysis of the committed SAE feature ablations: immediate drop, reconstruction artifact, dose invariance |
| `w1b_floor_measurement.py` | SSDC floors with the positional signal removed, APE and RoPE |
| `w1b_dinov3_rerun.py` | DINOv3 floor through `load_rope` on the native grid, merged into the W1-B output |
| `w1b_supplementary.py` | DINOv3 floor on the 224 grid; DINOv1 permutation-seed variance |
| `e12_bootstrap.py` | image-bootstrap SD of SSDC under RPI |
| `e12_seed_variance.py` | RPI permutation-seed SD of SSDC |
| `e12_dinov3_224.py` | both of the above for DINOv3 on the 224 grid |
| `w2_rotary_plane_masking.py` | all planes off, one axis off, random plane sets |
| `w2_perblock_gate.py` | checks for the per-block override that W3 uses |
| `w2_probe_protocol_ablation.py` | probe protocol variants on naver |
| `w3_window_masking.py` | prefix, suffix and middle windows; one-axis windows |
| `w3_rerun_ape_floor.py`, `w3_merge_ape_floor.py`, `w3_refresh_csvs.py` | APE floor re-run and merge into the W3 output |
| `w4_head_masking.py` | per-head carrying (stage A), top-k vs random (B), rebuilding (C), axis by head (D) |
| `w5_generality_extrapolation.py` | stages A to D on five checkpoints, 320 extrapolation, attention statistics |
| `w6_accuracy_and_noise.py` | rebuilder vs ordinary-head accuracy with paired SE, SSDC and probe noise, stage B draws |
| `w4_verify.py`, `w5_verify.py`, `w6_verify.py` | post-hoc checks recomputed from the JSON outputs |
| `w7_controls_rope.py` | every head's rotations off in blocks 6-11, in the intact model and after the 0-5 window, with per-image correctness; shuffled rotations against identity; probes by distance to the image border, with and without patch-to-prefix attention |
| `w7_controls_ape.py` | APE feature edits at block 2, with and without the SAE error: matched random latents, mean ablation, same-patch and other-patch swaps, embedding zeroed or permuted; probes at blocks 2, 6 and 11; DeiT-III baseline |
| `w7_merge_reruns.py` | merges the reg1-gap and SAM runs that machine restarts split into several files |
| `w7_tables.py` | CSV tables behind the paper's W7 figures and appendix tables, plus the patching tables from `causal_followups/` |

Rankings: carrying importance is the normalized SSDC loss at block 4 with one head's rotations off
in every block. Rebuilding importance is the SSDC loss at block 11 with rotations off in blocks 0-5
and one head's rotations held off in blocks 6-11, against the same window with all heads restored.

## Notes

- Images come from `experiments.common.load_imagenet`, which tries the gated `ILSVRC/imagenet-1k`
  split and falls back to the `benjamin-paine/imagenet-1k-256x256` repack. Every committed output
  from W1-A to W6 was produced from the repack. The repack is stored at 256×256, so the gated split gives slightly
  different numbers; to match the committed outputs, pass
  `dataset_id="benjamin-paine/imagenet-1k-256x256"` to `load_imagenet`.
- W1-A reads `results/runs/SAE_20k_images_analysis/original_results/feature_ablation_residual.json`
  and `second_seed/feature_ablation_residual_second_seed.json`. Its committed output was regenerated
  with this script and matches the original run in every number.
- W3: the first run lost the APE floor on a bookkeeping line (`range(None)` for a model with no
  rotary planes). `w3_window_masking.py` is the fixed version; the committed W3 output has the floor
  from `w3_rerun_ape_floor.py` merged in.
- W5 part E rebuilds q, k and v including rope and the separate q/k/v biases of reg1-gap, and gates
  the rebuild against the attention module's own output.
- W6 part A uses `predict_paired`, which is `predict()` plus per-image correctness, so arms scored
  on the same images get a paired SE.
- W7 loaded the gated `ILSVRC/imagenet-1k` split rather than the repack, so its absolute accuracies are
  2-6 points above W6's on the same arms. The effects agree: the top rebuilder costs 4.30, 5.74 and 7.62
  points after the 0-5 window on naver ViT-B, reg1-gap and naver ViT-S, against 4.40, 5.82 and 8.76 in W6.
- W7 file names carry a tag: `R12` for the all-heads and shuffled-rotation runs, `C3` for the border and
  prefix runs. The files the paper uses are `w7rope_R12_naver_2026-09-24T17-54-52Z.json`,
  `w7rope_R12_small_2026-09-24T22-49-53Z.json` (the 17-54-52 run stopped one condition short), the two
  `*_MERGED.json` files, and one file per model for everything else. Both W7 scripts save after every
  condition, and `W7_ONLY=<cond,...>` reruns named conditions. For reg1-gap and SAM a rerun overwrote the
  per-image files of the earlier arms, so paired errors exist only for the rerun conditions.
- SAM was later re-screened with a retrained SAE, which leaves 24 positional features instead of 33.
  `w7ape_sam_2026-09-25T18-42-10Z.json` is the W7 APE run with that set and is the SAM file the paper
  uses; `perimage/ape_sam_correct.npz` now holds its per-image correctness for every arm. The earlier
  SAM files (`w7ape_sam_2026-09-24*`, `2026-09-25T01*`, `T04*` and `w7ape_sam_MERGED.json`) used the
  33-feature set and are kept for the record.
- `w7_controls_ape.py` needs the SAE weights, which are not in the repository: set `W7_SAE_DIR` to the
  directory with `SAE_residual_APE_2_TOP64*.pt`. The positional feature lists are read from
  `results/runs/SAE_20k_images_analysis/`.
- Relative to the scripts as run, only these were changed: file names, the source-root path
  (now taken from the file location), a local `HF_HOME` default, the clone-fallback URL (now
  `None`, since the scripts sit inside the repo), W1-A's input paths, and some log wording. For W7:
  the file names (were `w7_reviewer_controls_*.py`), the APE script's location (moved into this
  folder, with its source-root path adjusted), the default SAE directory, and the header comments;
  the output folder was renamed to `w7/` and the run logs were dropped.
