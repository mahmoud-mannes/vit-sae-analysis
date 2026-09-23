"""Supplementary W1-B runs.

A) DINOv3 floor on the 224 grid, to match the other three models.
B) DINOv1 permutation-seed variance.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.ssdc import evaluate_ssdc
from main.load_models import load_rope, load_ape_timm
from experiments.common import load_imagenet

N, B, HALF = 1000, 256, torch.cuda.is_available()
DEV = "cuda" if HALF else "cpu"
SEED, SEEDS = 20260908, [20260908, 12345, 777, 2026]
DS = load_imagenet(split="validation", streaming=True)
out = {}

def ssdc(model, proc, src, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    s, _ = evaluate_ssdc(model, proc, DS, src, RPI=True, number_images=N,
                         batch_size=B, metric="manhattan", half=HALF, num_workers=0)
    return [float(x) for x in s]

def identity_like(e):
    sin, cos = e.chunk(2, -1)
    return torch.cat([torch.zeros_like(sin), torch.ones_like(cos)], -1)

class RopeIdentity:
    def __init__(s, m): s.mods = [(n, x) for n, x in m.named_modules() if callable(getattr(x, "get_embed", None))]
    def __enter__(s):
        s.orig = []
        for _, mod in s.mods:
            o = mod.get_embed; s.orig.append((mod, o))
            mod.get_embed = (lambda _o=o: (lambda *a, **k: identity_like(_o(*a, **k))))()
        return s
    def __exit__(s, *e):
        for mod, o in s.orig: mod.get_embed = o

# ---- A) DINOv3 at 224 ----
print("=" * 60 + "\nA) DINOv3 @ 224 (matched grid)", flush=True)
t0 = time.time()
m, p, src = load_rope(model_name="vit_base_patch16_dinov3.lvd1689m", device=DEV, half=HALF, input_size=224)
m.eval()
print(f"  transform input_size forced to 224; model grid {tuple(m.patch_embed.grid_size)} prefix {m.num_prefix_tokens}", flush=True)
base = ssdc(m, p, src)
with RopeIdentity(m): floor = ssdc(m, p, src)
L = 4
print(f"  block {L}: baseline {base[L]:+.4f}  floor {floor[L]:+.4f}  range {base[L]-floor[L]:+.4f}", flush=True)
out["RoPE · DINOv3 @224"] = {"probe_layer": L, "baseline_ssdc": base, "floor_ssdc": floor,
                             "achievable_range": base[L] - floor[L],
                             "grid": list(m.patch_embed.grid_size),
                             "n_prefix": int(m.num_prefix_tokens),
                             "native_256_baseline": 0.9462364, "native_256_floor": 0.0045862,
                             "runtime_s": round(time.time() - t0, 1)}
del m; torch.cuda.empty_cache()

# ---- B) DINOv1 permutation-seed variance ----
print("=" * 60 + "\nB) DINOv1 permutation-seed variance", flush=True)
t0 = time.time()
m, p, src = load_ape_timm(device=DEV, half=HALF); m.eval()
vals = []
for s in SEEDS:
    vals.append(ssdc(m, p, src, seed=s)[2])
    print(f"  seed {s:>8}: block 2 = {vals[-1]:+.4f}", flush=True)
a = np.array(vals)
out["APE · DINOv1 seed variance"] = {"probe_layer": 2, "seeds": SEEDS, "values": vals,
                                     "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                                     "range": float(a.max() - a.min()),
                                     "runtime_s": round(time.time() - t0, 1)}
print(f"  -> mean {a.mean():+.4f}  seed SD {a.std(ddof=1):.5f}  range {a.max()-a.min():.5f}", flush=True)

f = f"w1b_output/supplementary_{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}.json"
json.dump({"config": {"n_images": N, "batch": B, "seeds": SEEDS}, "results": out}, open(f, "w"), indent=1)
print("\nwrote", f, flush=True)
