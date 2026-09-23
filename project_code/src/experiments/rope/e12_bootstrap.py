"""E1.2: bootstrap SD of SSDC under RPI.

Keeps the per-image upper-triangle similarity vectors (SSDCAccumulator only keeps a
running sum) and resamples them offline. Baseline and floor see the same images in the
same order, so the bootstrap is paired.
"""
import os, sys, json, time, glob
from pathlib import Path
import numpy as np, torch
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics.ssdc import pairwise_cosine_similarity, _register_ssdc_accumulator_hooks
from main.load_models import get_vit_blocks, load_ape, load_rope, load_ape_timm
from main.model import predict
from main.prep_data import prep_data
from experiments.common import load_imagenet

N_IMAGES, BATCH, SEED, B_BOOT = 1000, 256, 20260908, 2000
HALF = torch.cuda.is_available(); DEV = "cuda" if HALF else "cpu"
DATASET = load_imagenet(split="validation", streaming=True)

MODELS = {
    "APE · google ViT": dict(load=lambda: load_ape(device=DEV, half=HALF),                                  family="APE",  probe=2),
    "RoPE · naver":     dict(load=lambda: load_rope(device=DEV, half=HALF),                                 family="RoPE", probe=4),
    "APE · DINOv1":     dict(load=lambda: load_ape_timm(device=DEV, half=HALF),                             family="APE",  probe=2),
    "RoPE · DINOv3":    dict(load=lambda: load_rope(model_name="vit_base_patch16_dinov3.lvd1689m",
                                                    device=DEV, half=HALF),                                 family="RoPE", probe=4),
}

class TriuRetainer:
    """Same add() contract as SSDCAccumulator, but keeps one triu vector per image."""
    def __init__(self, n_prefix):
        self.n_prefix = int(n_prefix); self._store = {}; self._iu = {}
    def add(self, key, tokens):
        tok = tokens.detach().float()[:, self.n_prefix:, :]
        S = pairwise_cosine_similarity(tok)                      # [B, T, T]
        T = S.shape[-1]
        if T not in self._iu:
            self._iu[T] = torch.triu_indices(T, T, offset=1, device=S.device)
        iu = self._iu[T]
        v = S[:, iu[0], iu[1]].half().cpu()                      # [B, P]
        self._store.setdefault(key, []).append(v)
    def stacked(self, key):
        return torch.cat(self._store[key], 0)                    # [N, P]
    def keys(self): return sorted(self._store)

def neg_dist_ranks(T):
    G = int(round(T ** 0.5)); assert G * G == T
    c = np.stack(np.meshgrid(np.arange(G), np.arange(G), indexing="ij"), -1).reshape(-1, 2)
    D = cdist(c, c, metric="cityblock")
    iu = np.triu_indices(T, k=1)
    return rankdata(-D[iu]), (-D[iu])

def run_condition(model, processor, source, n_prefix, ctx=None):
    acc = TriuRetainer(n_prefix)
    handles = _register_ssdc_accumulator_hooks(model, source, acc)
    try:
        torch.manual_seed(SEED); np.random.seed(SEED % (2**31))
        dl = prep_data(DATASET, processor, source, corruption_type=None,
                       number_images=N_IMAGES, batch_size=BATCH, half=HALF, num_workers=0)
        if ctx is None:
            predict(model, dl, source, RPI=True, half=HALF)
        else:
            with ctx:
                predict(model, dl, source, RPI=True, half=HALF)
    finally:
        for h in handles: h.remove()
    return acc

def bootstrap_block(M_base, M_floor, ranks_negD, idx):
    """idx: [B_BOOT, N] resample indices, shared between the two conditions (paired)."""
    out = {}
    rk = torch.as_tensor(ranks_negD, dtype=torch.float32, device=DEV)
    rk = (rk - rk.mean()) / rk.std()
    for name, M in (("baseline", M_base), ("floor", M_floor)):
        Mg = M.to(DEV, torch.float32)
        means = torch.stack([Mg[idx[b]].mean(0) for b in range(0, len(idx), 1)]) if False else None
        # chunked to bound memory
        vals = []
        for s in range(0, len(idx), 250):
            sel = idx[s:s+250]                              # [c, N]
            mm = Mg[sel].mean(1)                            # [c, P]
            r = mm.argsort(1).argsort(1).float()            # ordinal ranks
            r = (r - r.mean(1, keepdim=True)) / r.std(1, keepdim=True)
            vals.append((r * rk).mean(1))
        out[name] = torch.cat(vals).cpu().numpy()
        del Mg
        torch.cuda.empty_cache()
    return out

RESULTS = {}
for label, cfg in MODELS.items():
    print("\n" + "=" * 70 + f"\n{label}", flush=True)
    t0 = time.time()
    model, processor, source = cfg["load"](); model.eval()
    n_prefix = 1 if source == "transformers" else int(getattr(model, "num_prefix_tokens", 1))
    L = cfg["probe"]

    ctx = None
    if cfg["family"] == "RoPE":
        mods = [(n, m) for n, m in model.named_modules() if callable(getattr(m, "get_embed", None))]
        class _RO:
            def __enter__(s):
                s.orig = []
                for _, mod in mods:
                    o = mod.get_embed; s.orig.append((mod, o))
                    mod.get_embed = (lambda _o=o: (lambda *a, **k: (lambda e: torch.cat(
                        [torch.zeros_like(e.chunk(2, -1)[0]), torch.ones_like(e.chunk(2, -1)[1])], -1))(_o(*a, **k))))()
                return s
            def __exit__(s, *e):
                for mod, o in s.orig: mod.get_embed = o
        ctx = _RO()
    else:
        class _AZ:
            def __enter__(s):
                names = [n for n, _ in model.named_parameters()
                         if n.endswith("pos_embed") or n.endswith("position_embeddings")]
                name = sorted(names, key=len)[0]; parts = name.split(".")
                p = model
                for q in parts[:-1]: p = getattr(p, q)
                s.p, s.a = p, parts[-1]
                pe = getattr(p, s.a); s.saved = pe.detach().clone()
                with torch.no_grad(): pe.zero_()
                return s
            def __exit__(s, *e):
                with torch.no_grad(): getattr(s.p, s.a).copy_(s.saved)
        ctx = _AZ()

    acc_b = run_condition(model, processor, source, n_prefix, None)
    acc_f = run_condition(model, processor, source, n_prefix, ctx)
    del model; torch.cuda.empty_cache()

    P = acc_b.stacked(L).shape
    T = int((1 + (1 + 8 * P[1]) ** 0.5) / 2)
    ranks_negD, negD = neg_dist_ranks(T)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, P[0], (B_BOOT, P[0]), generator=g).to(DEV)

    entry = {"n_images": int(P[0]), "n_pairs": int(P[1]), "T": T, "probe_layer": L,
             "n_prefix": n_prefix, "blocks": {}}
    for blk in acc_b.keys():
        Mb, Mf = acc_b.stacked(blk), acc_f.stacked(blk)
        # sanity: full-weight SSDC must reproduce scipy on the plain mean
        full_b = float(spearmanr(negD, Mb.float().mean(0).numpy()).statistic)
        bo = bootstrap_block(Mb, Mf, ranks_negD, idx)
        d = bo["baseline"] - bo["floor"]
        zb, zf = np.arctanh(np.clip(bo["baseline"], -0.999999, 0.999999)), np.arctanh(np.clip(bo["floor"], -0.999999, 0.999999))
        entry["blocks"][int(blk)] = {
            "ssdc_point_scipy": full_b,
            "baseline_mean": float(bo["baseline"].mean()), "baseline_sd": float(bo["baseline"].std()),
            "baseline_ci": [float(np.percentile(bo["baseline"], 2.5)), float(np.percentile(bo["baseline"], 97.5))],
            "floor_mean": float(bo["floor"].mean()), "floor_sd": float(bo["floor"].std()),
            "delta_mean": float(d.mean()), "delta_sd": float(d.std()),
            "z_baseline_sd": float(zb.std()), "z_delta_sd": float((zb - zf).std()),
        }
        if int(blk) == L:
            e = entry["blocks"][int(blk)]
            print(f"  block {blk}: scipy {full_b:+.4f} | boot {e['baseline_mean']:+.4f} "
                  f"± {e['baseline_sd']:.4f}  CI [{e['baseline_ci'][0]:+.4f}, {e['baseline_ci'][1]:+.4f}]"
                  f" | z sd {e['z_baseline_sd']:.4f}", flush=True)
    entry["runtime_s"] = round(time.time() - t0, 1)
    RESULTS[label] = entry
    print(f"  done in {entry['runtime_s']}s", flush=True)
    del acc_b, acc_f

out = f"w1b_output/e12_bootstrap_{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}.json"
json.dump({"config": {"n_images": N_IMAGES, "batch": BATCH, "seed": SEED, "n_boot": B_BOOT},
           "results": RESULTS}, open(out, "w"), indent=1)
print("\nwrote", out)
