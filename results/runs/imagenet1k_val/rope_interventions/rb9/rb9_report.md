# RB-9 results — generated 2026-11-10T14-57-07Z

Rules (fixed before the run):

- Erasure check at an edited site (its readout sees the edited stream): normalised SSDC N <= 0.1, and on held-out images exact probe <= 4.0x chance and row and column probes <= 1.5x chance (RB-6 / RB-8 thresholds). Where N is undefined (intact - floor < 0.10) the probes decide.
- N = (SSDC - floor) / (intact - floor) at the same readout, only where intact - floor >= 0.1. Floor = all rotations off (RoPE), embedding zeroed (APE), both (hybrid).
- Classifiers (RB-8's rule, unchanged). C = top-1 cost of full_mean (every site), C0 = cost of full_offset (the same vectors at the wrong slots), F = cost of removing position entirely. BARELY USED if C < 10% of F; USED if C >= 10% of F and C - C0 > 3 paired SE; CANNOT TELL otherwise. The statement covers the per-slot mean component only.
- Outcome (labels only; the team decides the text): A if every RoPE classifier (naver ViT-B, naver ViT-S, reg1-gap) reads BARELY USED; B if any reads USED; C otherwise. The hybrid and the APE models are reported with the same rule.

**Outcome: A** (naver ViT-B: BARELY USED, naver ViT-S: BARELY USED, reg1-gap ViT-B: BARELY USED)

## Top-1 (points ± paired SE, 5,000 ILSVRC images)

| model | intact | F (full removal) | RB-8 edit (block inputs only) | RB-9 edit (every site) | % of F | control (wrong slots) | RB-9 − RB-8 | verdict |
|---|---|---|---|---|---|---|---|---|
| naver ViT-B | 84.64 | 45.80 | 0.28 ± 0.21 | 0.42 ± 0.21 | 0.9% | 0.16 ± 0.15 | 0.14 ± 0.10 | BARELY USED |
| naver ViT-S | 81.36 | 57.30 | 0.62 ± 0.23 | 0.64 ± 0.24 | 1.1% | 0.36 ± 0.19 | 0.02 ± 0.17 | BARELY USED |
| reg1-gap ViT-B | 83.88 | 54.38 | 0.52 ± 0.20 | 0.30 ± 0.21 | 0.6% | 0.28 ± 0.16 | −0.22 ± 0.14 | BARELY USED |
| hybrid ViT-B | 84.82 | 51.40 | 0.64 ± 0.22 | 0.62 ± 0.22 | 1.2% | 0.30 ± 0.19 | −0.02 ± 0.07 | BARELY USED |
| supervised ViT-B | 81.16 | 57.04 | 53.22 ± 0.74 | 53.00 ± 0.75 | 92.9% | 15.58 ± 0.61 | −0.22 ± 0.12 | USED |
| AugReg ViT-B | 80.14 | 50.92 | 48.56 ± 0.74 | 48.60 ± 0.74 | 95.4% | 8.26 ± 0.47 | 0.04 ± 0.06 | USED |
| SAM ViT-B | 81.24 | 42.10 | 37.36 ± 0.72 | 37.34 ± 0.72 | 88.7% | 5.16 ± 0.40 | −0.02 ± 0.03 | USED |
| DeiT-III ViT-B | 84.58 | 61.06 | 57.74 ± 0.72 | 57.72 ± 0.72 | 94.5% | 14.62 ± 0.57 | −0.02 ± 0.05 | USED |

## The erasure at every site (normalised SSDC; probes on held-out images, chance 0.51% exact)

| model | sites | sites meeting the erasure check | exact probe over the sites | intact exact probe, max over the same readouts | N entering block 11: RB-8 → RB-9 | N at MLP 11 (RB-9) | N at the final output: RB-8 → RB-9 |
|---|---|---|---|---|---|---|---|
| naver ViT-B | 24 | 3 | 0.9–3.0% | 37.2% | 0.86 → 0.85 | 0.83 | 1.00 → 0.95 |
| naver ViT-S | 24 | 5 | 0.7–2.5% | 26.8% | 0.98 → 0.92 | 0.82 | 0.91 → 0.89 |
| reg1-gap ViT-B | 24 | 3 | 0.8–2.8% | 39.8% | 0.62 → 0.69 | 0.75 | 0.80 → 0.78 |
| hybrid ViT-B | 25 | 8 | 0.6–2.3% | 99.5% | 0.85 → 0.84 | 0.78 | 0.81 → 0.78 |
| DINOv3 ViT-B | 24 | 2 | 0.8–4.3% | 44.5% | 0.17 → 0.18 | 0.28 | 0.25 → 0.25 |
| supervised ViT-B | 25 | 25 | 0.6–1.7% | 100.0% | 0.07 → 0.04 | — | — → — |
| AugReg ViT-B | 25 | 25 | 0.6–0.8% | 100.0% | −0.02 → −0.02 | −0.03 | −0.02 → −0.03 |
| SAM ViT-B | 25 | 25 | 0.6–0.9% | 100.0% | 0.00 → 0.00 | 0.01 | 0.01 → 0.01 |
| DeiT-III ViT-B | 25 | 25 | 0.7–1.1% | 99.9% | 0.01 → 0.01 | 0.01 | 0.00 → 0.00 |
| DINO ViT-B | 25 | 25 | 0.7–0.9% | 100.0% | 0.01 → 0.01 | −0.00 | −0.00 → −0.01 |

## Text for the paper (by the outcome above)

- M17d, last sentence: “Subtracting it before the attention and before the MLP of every block at once costs the RoPE classifiers 0.3–0.6 top-1 points (0.6–1.1% of removing every rotation) and APE 37.3–57.7 (89–95% of removing the embedding); it removes most decodable position but not the RoPE classifiers’ SSDC (Appendix G.3).”
- G.3, the paragraph on the erasure at every block: describe the RB-9 edit (both sites of every block, plus the last block’s output) with the numbers in the two tables above, and add: “Editing only the input of each block, which leaves the position that attention writes within a block visible to that block’s MLP, gives the same result (RoPE classifiers 0.28–0.62 points).”
- Table 27: replace the cost and probe columns with RB-9’s, keep RB-8’s cost as one extra column.

Gates: every gate of every session is in rb9_gate_report.csv (written by the last cell); the run is valid only if it shows 0 FAIL and 0 SKIP.