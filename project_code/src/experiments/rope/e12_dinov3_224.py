"""E1.2 for DINOv3 on the 224 grid, the grid used by the reference ablation runs
(load_rope with input_size=224).
"""
import os, sys, json, time, glob
from pathlib import Path
import numpy as np, torch
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.ssdc import pairwise_cosine_similarity, _register_ssdc_accumulator_hooks, evaluate_ssdc
from main.load_models import load_rope
from main.prep_data import prep_data
from main.model import predict
from experiments.common import load_imagenet

N, B, SEED, NBOOT = 1000, 256, 20260908, 2000
SEEDS = [20260908, 12345, 777, 2026]
HALF = torch.cuda.is_available(); DEV = "cuda" if HALF else "cpu"
DS = load_imagenet(split="validation", streaming=True); L = 4

class TriuRetainer:
    def __init__(s, n): s.n = int(n); s._st = {}; s._iu = {}
    def add(s, k, toks):
        t = toks.detach().float()[:, s.n:, :]
        S = pairwise_cosine_similarity(t); T = S.shape[-1]
        if T not in s._iu: s._iu[T] = torch.triu_indices(T, T, offset=1, device=S.device)
        iu = s._iu[T]; s._st.setdefault(k, []).append(S[:, iu[0], iu[1]].half().cpu())
    def stacked(s, k): return torch.cat(s._st[k], 0)

def identity_like(e):
    sin, cos = e.chunk(2, -1)
    return torch.cat([torch.zeros_like(sin), torch.ones_like(cos)], -1)

m, p, src = load_rope(model_name="vit_base_patch16_dinov3.lvd1689m", device=DEV, half=HALF, input_size=224)
m.eval(); npref = int(m.num_prefix_tokens)

acc = TriuRetainer(npref)
h = _register_ssdc_accumulator_hooks(m, src, acc)
torch.manual_seed(SEED); np.random.seed(SEED % (2**31))
dl = prep_data(DS, p, src, corruption_type=None, number_images=N, batch_size=B, half=HALF, num_workers=0)
predict(m, dl, src, RPI=True, half=HALF)
for x in h: x.remove()

M = acc.stacked(L); Nimg, P = M.shape
T = int((1 + (1 + 8 * P) ** 0.5) / 2); G = int(round(T ** 0.5))
print(f"grid {G}x{G}, patch tokens {T}, prefix {npref}, pairs {P}", flush=True)

co = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
D = cdist(co, co, metric="cityblock"); iu = np.triu_indices(T, k=1)
negD = -D[iu]; rk = torch.as_tensor(rankdata(negD), dtype=torch.float32, device=DEV)
rk = (rk - rk.mean()) / rk.std()
print("point SSDC (scipy):", f"{spearmanr(negD, M.float().mean(0).numpy()).statistic:+.6f}", flush=True)

g = torch.Generator().manual_seed(SEED)
idx = torch.randint(0, Nimg, (NBOOT, Nimg), generator=g).to(DEV)
Mg = M.to(DEV, torch.float32); vals = []
for s in range(0, NBOOT, 250):
    mm = Mg[idx[s:s+250]].mean(1)
    r = mm.argsort(1).argsort(1).float()
    r = (r - r.mean(1, keepdim=True)) / r.std(1, keepdim=True)
    vals.append((r * rk).mean(1))
boot = torch.cat(vals).cpu().numpy()
del Mg; torch.cuda.empty_cache()

seedvals = []
for s in SEEDS:
    torch.manual_seed(s); np.random.seed(s % (2**31))
    sc, _ = evaluate_ssdc(m, p, DS, src, RPI=True, number_images=N, batch_size=B,
                          metric="manhattan", half=HALF, num_workers=0)
    seedvals.append(float(sc[L])); print(f"  seed {s}: {seedvals[-1]:+.4f}", flush=True)

a = np.array(seedvals); zb = np.arctanh(np.clip(boot, -0.999999, 0.999999))
out = {"grid": [G, G], "patch_tokens": T, "n_prefix": npref, "probe_layer": L,
       "bootstrap_mean": float(boot.mean()), "bootstrap_sd": float(boot.std()),
       "bootstrap_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
       "z_baseline_sd": float(zb.std()),
       "seeds": SEEDS, "seed_values": seedvals, "seed_sd": float(a.std(ddof=1)),
       "combined_sd": float(np.hypot(boot.std(), a.std(ddof=1)))}
print(f"\nbootstrap SD {out['bootstrap_sd']:.5f} | seed SD {out['seed_sd']:.5f} | combined {out['combined_sd']:.5f} | z SD {out['z_baseline_sd']:.5f}")
f = f"w1b_output/e12_dinov3_224_{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}.json"
json.dump(out, open(f, "w"), indent=1); print("wrote", f)
