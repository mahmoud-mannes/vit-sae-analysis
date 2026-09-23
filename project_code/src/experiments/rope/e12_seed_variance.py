"""Seed variance of SSDC under RPI.

Repeats one condition under different RPI permutation seeds, to separate permutation
noise from the image-sampling noise measured by e12_bootstrap.py.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.ssdc import evaluate_ssdc
from main.load_models import load_rope, load_ape
from experiments.common import load_imagenet

N, B, HALF = 1000, 256, torch.cuda.is_available()
DEV = "cuda" if HALF else "cpu"
SEEDS = [20260908, 12345, 777, 2026]
DATASET = load_imagenet(split="validation", streaming=True)

CASES = {
    "RoPE · DINOv3": (lambda: load_rope(model_name="vit_base_patch16_dinov3.lvd1689m", device=DEV, half=HALF), 4),
    "RoPE · naver":  (lambda: load_rope(device=DEV, half=HALF), 4),
    "APE · google ViT": (lambda: load_ape(device=DEV, half=HALF), 2),
}

out = {}
for label, (loader, L) in CASES.items():
    model, processor, source = loader(); model.eval()
    vals = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s % (2**31))
        sc, _ = evaluate_ssdc(model, processor, DATASET, source, RPI=True,
                              number_images=N, batch_size=B, metric="manhattan",
                              half=HALF, num_workers=0)
        vals.append(float(sc[L]))
        print(f"  {label} seed {s:>8}: block {L} = {vals[-1]:+.4f}", flush=True)
    a = np.array(vals)
    out[label] = {"probe_layer": L, "seeds": SEEDS, "values": vals,
                  "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                  "range": float(a.max() - a.min())}
    print(f"  -> mean {a.mean():+.4f}  seed SD {a.std(ddof=1):.4f}  range {a.max()-a.min():.4f}\n", flush=True)
    del model; torch.cuda.empty_cache()

p = f"w1b_output/seed_variance_{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}.json"
json.dump({"config": {"n_images": N, "batch": B, "seeds": SEEDS}, "results": out}, open(p, "w"), indent=1)
print("wrote", p)
