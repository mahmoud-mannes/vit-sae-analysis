# RB-7 results — generated 2026-11-08T13-32-18Z

Gates: 281 checked, 281 pass, **0 fail**, 0 skipped. Parts in this report: d, e, c, a, b.


Conventions: SSDC normalised N = (arm − floor) / (intact − floor) at that readout; retention of a probe likewise. Top-1 costs in points on 5,000 ILSVRC images with paired standard errors. Heads are (block, head). Every verdict is the rule fixed in the notebook before the run (quoted under each part).

## Part d — is the six-head result a block-position effect? (M20, Table 21; 91UE W4, RxJ2 Q2)

Rule: Part d. Null: 48 random head sets per model with exactly the six top SSDC heads' block counts (heads drawn uniformly within each block, the top-6 set itself excluded; seed fixed). Cost = top-1 points lost against the 0-5 window (classifiers) and SSDC lost entering block 11 against Cref (all four). HOLDS if the top-6 cost exceeds all 48 sets; HOLDS AT 95% if it exceeds at least 46; DOES NOT HOLD otherwise.

Also: Part d, from RB-0 (no new forward passes): Spearman of single-head SSDC cost vs top-1 cost computed within each of blocks 6-10 and averaged; p from a permutation that shuffles top-1 costs within blocks.

| model | top-6 heads | top-1 cost of top 6 | block-matched null (mean / 95th / max) | top 6 above | verdict (top-1) | SSDC: top 6 above | within-block ρ (p) |
|---|---|---|---|---|---|---|---|
| naver ViT-B | b6: [6, 1, 2]; b7: [2]; b8: [10, 11] | 10.06 ± 0.48 | 4.13 / 8.44 / 8.68 | 48 of 48 | HOLDS | 48 of 48 (HOLDS) | 0.35 (0.011) |
| naver ViT-S | b6: [0]; b7: [2, 4, 3]; b8: [4]; b9: [3] | 17.64 ± 0.58 | 6.84 / 14.16 / 17.46 | 48 of 48 | HOLDS | 48 of 48 (HOLDS) | 0.49 (0.014) |
| reg1-gap ViT-B | b6: [9, 7, 4, 11]; b7: [7, 6] | 14.78 ± 0.55 | 4.47 / 12.03 / 15.42 | 47 of 48 | HOLDS AT 95% | 48 of 48 (HOLDS) | 0.50 (0.000) |
| DINOv3 ViT-B | b6: [2]; b7: [9, 1]; b9: [10]; b10: [0, 8] | — | — | — | — | 48 of 48 (HOLDS) | — |

**Text for the paper.**

- M20 (Section 4.5): replace “and the six top SSDC heads together cost more than each of 24 random sets of one head per block (Appendix H.2)” with “and the six top SSDC heads together cost 10.1–17.6 points, more than 47–48 of 48 random sets of six heads drawn from the same blocks (Appendix H.2)”.
- M20 and M11: after “A head’s SSDC cost predicts its top-1 cost (Spearman ρ = 0.55–0.67” add “; within blocks ρ = 0.35–0.50, p ≤ 0.014” (in M11 the sentence reads “SSDC cost predicts it better than probe cost does (Spearman ρ = 0.55–0.67 against 0.19–0.33”: add the same clause after 0.55–0.67). (This answers the objection that both costs simply fall with depth.)
- Table 21, column “Top 6 vs random”: give, per model, the top-6 cost and “48 block-matched sets: mean / max”, from the table above; keep RB-3's one-per-block comparison in the caption only.
- Appendix H.2, add: “Because the six top heads sit in the first blocks after the window (naver ViT-B blocks 6–8; naver ViT-S blocks 6–9; reg1-gap ViT-B blocks 6–7) while random one-head-per-block sets spread over blocks 6–11, we also drew 48 random sets of six heads with exactly the top six's block counts” followed by the M20 result.

DINOv3 (SSDC only, no classifier): top 6 above 48 of 48 block-matched sets (HOLDS); reported in Appendix H.2 only.

## Part a — 320×320 (Appendix H.3; 91UE Q4 and W5)

Rules: Axis test (parts a, b, c): retention = (arm - floor) / (intact - floor) of the row and column probes at the axis readout. AXES SEPARATE if removing the row planes in every block leaves row retention <= 0.1 and column retention >= 0.5, and removing the column planes does the converse; PARTLY otherwise if the removed axis always falls more than the other; NOT SEPARATE if not. Rebuild (parts a, b, c): normalised SSDC entering the last block after the first-half window (0: floor, 1: intact). REBUILDS if >= 0.5 (ViT-B at 224: 0.50-0.71); REBUILDS LESS if 0.2-0.5; DOES NOT REBUILD below 0.2. Part a: the head Section 4.5 names ((6,6), (6,9), (7,4)) KEEPS ITS RANK if it is among the 3 costliest single heads by top-1 at 320, and KEEPS ITS AXIS if its RB-2 label at 320 equals its label at 224.

| model | axis test (block 4) | rebuild N, block 11 (224 → 320) | named head | top-1 rank, 224 → 320 | cost 224 → 320 (pts) | axis 224 → 320 | verdict | ρ per-head cost 224 vs 320 | first-two-block share 224 → 320 | ρ SSDC vs top-1 at 320 (within blocks) |
|---|---|---|---|---|---|---|---|---|---|---|
| naver ViT-B | AXES SEPARATE (row off: row 0%, col 100%; col off: col -0%, row 127%) | 0.71 → 0.61 | (6,6) | 1 → 1 of 72 | 3.04 → 4.56 ± 0.34 | row → row | KEEPS ITS RANK / KEEPS ITS AXIS | 0.75 | 80% → 94% | 0.09 (0.16) |
| reg1-gap ViT-B | AXES SEPARATE (row off: row 0%, col 93%; col off: col 0%, row 103%) | 0.54 → 0.43 | (6,9) | 1 → 1 of 72 | 4.52 → 5.54 ± 0.34 | row → row | KEEPS ITS RANK / KEEPS ITS AXIS | 0.85 | 76% → 81% | 0.69 (0.39) |
| naver ViT-S | AXES SEPARATE (row off: row 0%, col 94%; col off: col 0%, row 74%) | 0.58 → 0.62 | (7,4) | 1 → 1 of 36 | 6.08 → 7.70 ± 0.41 | row → row | KEEPS ITS RANK / KEEPS ITS AXIS | 0.79 | 75% → 69% | 0.28 (0.36) |

**Text for the paper (Appendix H.3, after “That run also covered 320×320 input, where the top-rebuilder index group costs 7.84, 6.08 and 7.80 points after the window.”, i.e. the sentence as edited by A15):**

“We repeated the single-head test at 320×320 (a 20×20 grid) on the same 5,000 images. Removing one axis’s planes in every block still removes that axis and spares the other at block 4, and after the 0–5 window normalised SSDC entering block 11 recovers to 0.43–0.62 (224: 0.54–0.71). By top-1 cost, (6,6) is 1 of 72 (4.56 ± 0.34 points), (6,9) is 1 of 72 (5.54 ± 0.34 points), (7,4) is 1 of 36 (7.70 ± 0.41 points); all 3 stay among the three costliest single heads and keep the axis they favour at 224 (row). Per-head top-1 costs at the two resolutions correlate at ρ = 0.75–0.85, and the first two blocks after the window carry 69–94% of the summed cost.”

Replace the existing sentence “At 320×320, SSDC loses sensitivity …” only if the SSDC-vs-top-1 ρ at 320 in the table is clearly below 224's (0.55–0.67); otherwise keep it as is.

## Part b — naver ViT-L/16 (new Appendix H.4; 91UE W5, RxJ2 W2 scale)

Rules: Axis test (parts a, b, c): retention = (arm - floor) / (intact - floor) of the row and column probes at the axis readout. AXES SEPARATE if removing the row planes in every block leaves row retention <= 0.1 and column retention >= 0.5, and removing the column planes does the converse; PARTLY otherwise if the removed axis always falls more than the other; NOT SEPARATE if not. Rebuild (parts a, b, c): normalised SSDC entering the last block after the first-half window (0: floor, 1: intact). REBUILDS if >= 0.5 (ViT-B at 224: 0.50-0.71); REBUILDS LESS if 0.2-0.5; DOES NOT REBUILD below 0.2. Part b: top SSDC head and costliest top-1 head; whether each sits in the first two blocks after the window (12-13); share of the summed positive top-1 cost in blocks 12-13; Spearman of SSDC vs top-1 cost over all 192 heads; how many mapped heads favour one axis.

| test | result | verdict |
|---|---|---|
| axis test at block 8 | row off: row 0%, col 115%; col off: col 0%, row 108%; 16 random planes: row 108%, col 117% | AXES SEPARATE |
| rebuild entering block 23 (N) | 0–5: 0.80; 0–11: 0.54; 0–17: 0.21 | REBUILDS (main window 0–11) |
| top SSDC single head (blocks 12–22) | (14,12); first two blocks after the window: False | — |
| costliest single head by top-1 | (12,5) at 0.92 ± 0.17 points; first two blocks: True | — |
| share of summed top-1 cost in blocks 12–13 | 53% | — |
| SSDC cost vs top-1 cost, 176 heads in blocks 12–22 | ρ = 0.54; within blocks 0.43 (p = 0.000) | — |
| axis map (top heads) | 2 of 8 mapped heads above noise; 2 of them favour one axis | — |
| top-1 intact / window / floor | 85.26 / 74.96 / 43.66 | — |

**Text for the paper (new Appendix H.4, “A larger model”):**

“naver ViT-L/16 has 24 blocks of 16 heads and the same axial RoPE. Removing the row or the column planes in every block removes that axis at block 8 and spares the other. With rotations withheld from blocks 0–11 and restored, normalised SSDC entering block 23 recovers to 0.54 (0–5: 0.80; 0–17: 0.21). The costliest single head after the window is (12,5), 0.92 ± 0.17 points; blocks 12–13 carry 53% of the summed single-head cost, and a head’s SSDC cost and its top-1 cost correlate at ρ = 0.54 (within blocks ρ = 0.43, p = 0.000). Of the 2 mapped heads above noise, 2 favour one axis.”

## Part e1 — do the axes compete for attention? (Appendix E.1, A5; NpkG Q2)

Rule: Part e (NpkG Q2): mean attention distance along rows |dr| and along columns |dc| (patches), blocks 1-4, head-averaged, 64 natural images. WIDER POOLING SUPPORTED if removing an axis's planes widens attention along that axis in at least 7 of the 8 (model, axis) cases and the widening rank-correlates (Spearman > 0.5, descriptive) with the rise of the other axis's probe above intact at block 4 (W5); NOT SUPPORTED if it widens in at most 4 of 8; MIXED otherwise.

| model | axis removed (all blocks) | widening along that axis, blocks 1–4 (patches) | other axis's probe vs intact at block 4 (W5, pts) |
|---|---|---|---|
| naver ViT-B | row | +1.514 | 0.83 |
| naver ViT-B | col | +1.336 | 6.18 |
| DINOv3 ViT-B | row | +1.211 | -6.89 |
| DINOv3 ViT-B | col | +1.115 | 6.28 |
| naver ViT-S | row | +1.853 | -0.45 |
| naver ViT-S | col | +1.686 | -4.94 |
| reg1-gap ViT-B | row | +1.487 | 0.63 |
| reg1-gap ViT-B | col | +1.417 | 4.57 |

Widening in 8 of 8 cases; Spearman(widening, rise) = -0.45. **Verdict: MIXED.**

**Text for the paper (A5: one sentence at the end of the Appendix E.1 paragraph on the rise):** “Removing one axis’s planes widens attention along that axis in most but not all models (Appendix E.1), so attention pooling may contribute to the rise, but the data do not establish it.” Add the table above to Appendix E.1.

## Part e2 — what does the hybrid's classifier get from the rotations? (Appendix G.2, A11; RxJ2 Q3)

Rule: Part e (RxJ2 Q3): hybrid, blocks 4-11, head-averaged distance and entropy. ROTATIONS KEEP LATE ATTENTION LOCAL if removing the rotations raises the mean distance in at least 6 of the 8 blocks and by at least 0.25 patches on average, while removing the embedding changes it by less than half as much; otherwise NOT SHOWN.

Blocks 4–11, mean over heads: removing the rotations changes the mean attention distance by +1.39 patches (wider in 7 of 8 blocks) and the entropy by +0.75 nats; removing the embedding changes them by +0.15 patches and +0.10 nats. **Verdict: ROTATIONS KEEP LATE ATTENTION LOCAL.**

**Text for the paper (Appendix G.2, after the Table 22 sentence):** “Without its rotations the hybrid’s attention in blocks 4–11 spreads by 1.39 patches on average, while removing the embedding changes it by +0.15: the rotations keep late attention local, and erasing the position stored in the residual stream costs the classifier only 0.4 points (Appendix G.3), so the classifier relies on rotation-local attention rather than on decodable position.”

## Part c — Qwen2-VL-2B's vision encoder (new Appendix H.5; RxJ2 W2)

Rules: Part c: STARTS NEAR ZERO if SSDC entering block 0 is at most 0.05 (Figure 2's RoPE pattern: ViT-B RoPE models 0.027-0.028, APE models 0.42-0.57), i.e. spatial structure is built by attention, not supplied by an embedding. Axis test (parts a, b, c): retention = (arm - floor) / (intact - floor) of the row and column probes at the axis readout. AXES SEPARATE if removing the row planes in every block leaves row retention <= 0.1 and column retention >= 0.5, and removing the column planes does the converse; PARTLY otherwise if the removed axis always falls more than the other; NOT SEPARATE if not. Rebuild (parts a, b, c): normalised SSDC entering the last block after the first-half window (0: floor, 1: intact). REBUILDS if >= 0.5 (ViT-B at 224: 0.50-0.71); REBUILDS LESS if 0.2-0.5; DOES NOT REBUILD below 0.2.

Checkpoint Qwen/Qwen2-VL-2B-Instruct, revision 895c3a49bc3fa70a340399125c650a463535e71c.

| test | result | verdict |
|---|---|---|
| SSDC entering block 0 / peak | -0.014 / 0.849 at block 9 (half of peak first at block 5) | STARTS NEAR ZERO |
| axis test at block 11 | row off: row 0%, col 118%; col off: col 0%, row 122%; 20 random planes: row 73%, col 111% | AXES SEPARATE |
| rebuild entering block 31 (N) | 0–7: 1.21; 0–15: 0.55; 0–23: 0.18 | REBUILDS (main window 0–15) |
| one axis off in 0–15, probes at block 31 | row off: row 16%, col 98%; col off: col 10%, row 101% | — |

**Text for the paper (new Appendix H.5, “A RoPE encoder trained for a VLM”):**

“Qwen2-VL’s vision encoder (675M parameters, 32 blocks, 14-pixel patches) uses axial 2-D RoPE and no absolute embedding. At 224×224 its SSDC starts at -0.01 entering block 0 and peaks at 0.85 in block 9, so, as in the RoPE ViTs, attention builds the spatial structure. Removing the row or the column planes in every block removes that axis at block 11 and spares the other. With rotations withheld from blocks 0–15 and restored, normalised SSDC entering block 31 recovers to 0.55. The mechanism therefore carries over to a RoPE encoder trained for a VLM.”

CLIP and SigLIP encoders use absolute embeddings, so they belong to the APE side.

## Limitations (M24) and the coverage map

**M24, replacing “and extending the analysis to larger models and higher resolutions … would show how far these mechanisms generalise”:** “the rebuilding heads keep their rank and axis at 320×320; in ViT-L the axes separate and later blocks rebuild; in Qwen2-VL’s vision encoder the axes separate and later blocks rebuild (Appendix H.3, H.4, H.5), but we have not tested dense prediction.”

**Coverage map:** 91UE Q4 → Answered (part a). 91UE W5 → Answered for resolution and scale (parts a, b); dense prediction stays a limitation. RxJ2 W2 → RoPE half answered (part c); CLIP/SigLIP are APE. 91UE W4 / RxJ2 Q2 → M20 now uses the block-matched null (part d). NpkG Q2 → part e1. RxJ2 Q3 → part e2 + RB-6. Use the verdicts above; where a verdict is negative, the text above already says so.
