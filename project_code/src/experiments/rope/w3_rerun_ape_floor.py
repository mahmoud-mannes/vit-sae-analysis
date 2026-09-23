"""Re-measure the floor for the two APE models.

The main W3 run dropped it on a bookkeeping line (range(None) for a model without
rotary planes). Only the floor condition is affected.
"""
import json, glob, os, time
from pathlib import Path
import torch

SRC = open(Path(__file__).with_name("w3_window_masking.py")).read()
g = {"__name__": "__w3_defs__"}
exec(compile(SRC[:SRC.index("# ---- cell 19 ----")], "w3_window_masking.py", "exec"), g)

MODELS, LOADERS, ANCHORS = g["MODELS"], g["LOADERS"], g["W2_ANCHORS"]
DEVICE, N_BLOCKS = g["DEVICE"], g["N_BLOCKS"]

patch = {}
for label, cfg in MODELS.items():
    if cfg["family"] == "RoPE":
        continue
    print("\n" + "=" * 70); print(label); print("=" * 70)
    t0 = time.time()
    model, processor, source = LOADERS[cfg["loader"]](device=DEVICE, half=False, **cfg["kwargs"])
    model.eval().float()

    n_prefix = 1 if source == "transformers" else int(getattr(model, "num_prefix_tokens", 1))
    hw = g["capture_forward_grid"](model, processor, source)
    if hw is None:
        raise RuntimeError("could not read the forward-pass grid")
    H, W = hw
    g["gate"]("forward grid is 14x14", label, (H, W) == (14, 14), f"{H}x{W}")

    conds, _ = g["build_conditions"](cfg["family"], None, [], [], {})
    spec = next(c for c in conds if c["name"] == "floor")
    ctx = g["make_ctx"](model, source, spec, cfg["family"], None, None)

    r = g["measure"](model, processor, source, ctx, spec["probe_blocks"],
                     n_prefix, H, W, "floor")
    r.update({k: spec[k] for k in ("kind", "window", "read_from", "probe_blocks")})
    r["planes_killed"] = []
    patch[label] = r

    PL = ANCHORS[label]["probe_layer"]
    print(f"  floor ssdc@{PL}={r['ssdc'][PL]:+.6f}  anchor {ANCHORS[label]['floor']}  {r['runtime_s']}s")
    g["gate"]("floor hits its W1-B anchor", label,
              abs(r["ssdc"][PL] - ANCHORS[label]["floor"]) < g["ANCHOR_TOL"],
              f"{r['ssdc'][PL]:+.6f} vs anchor {ANCHORS[label]['floor']}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  model done in {time.time()-t0:.1f}s")

out = "w3_output/ape_floor_patch.json"
json.dump({"generated_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime()),
           "reason": "main run lost these to a range(None) on the planes_killed line",
           "conditions": patch}, open(out, "w"), indent=1)
print("\nwritten:", out)
G=g["GATES"]
fails=[x for x in G if x["status"]=="FAIL"]
print(f"\ngates: {len(G)-len(fails)} PASS, {len(fails)} FAIL")
for x in fails: print("  FAIL", x["scope"], x["gate"], x["detail"])
