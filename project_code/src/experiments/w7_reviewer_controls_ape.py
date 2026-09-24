"""W7 reviewer controls, APE side (B1, B7-lite, C4, B8, C1-lite).

For each APE model, at the residual stream entering block 2, with the published SAE and
the published positional feature set (full set: 20 / 18 / 20 / 33 latents):
  recon             SAE reconstruction, nothing removed (published baseline)
  pos_zero          positional latents zeroed (published intervention)
  rand_uniform      same number of latents drawn uniformly from the dictionary (published control)
  rand_matched_{k}  latents matched to the positional ones on firing rate and activation mass (5 draws)
  *_err             same edits with the SAE error term added back: x + D(z') - D(z)
  pos_mean_err      positional latents replaced by their mean activation
  pos_resample_samepos_err   positional latents taken from another image, same position
  pos_resample_diffpos_err   positional latents taken from another position, same image
  embed_zero / embed_perm    no SAE: learned position embedding zeroed / permuted over positions
Readouts: top-1 with per-image correctness (models with a head), SSDC per block under RPI,
exact/row/col probes at blocks 2, 6, 11 with a train/val/test split (epoch chosen on val).
DeiT-III (recipe-matched to the naver RoPE ViTs) gets the no-SAE conditions only (C1-lite).
"""
import os, sys, json, time, math, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from metrics.ssdc import evaluate_ssdc
from main.load_models import get_vit_blocks, get_block_attention, get_patch_embed_conv, load_ape, load_ape_timm
from main.prep_data import prep_data
from experiments.common import load_imagenet
from SAE.sae import SAE
from SAE_causal.feature_ablation import load_top_features

OUT = Path(os.environ.get("W7_OUT", "w7_output")); OUT.mkdir(parents=True, exist_ok=True)
PER = OUT / "perimage"; PER.mkdir(exist_ok=True)
STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
DEV = "cuda"
BATCH = 128
N_SS = int(os.environ.get("W7_N_SS", 1000))
N_PROBE = int(os.environ.get("W7_N_PROBE", 1000))
N_STATS = int(os.environ.get("W7_N_STATS", 200))
PROBE_BLOCKS = [2, 6, 11]
SAE_DIR = Path(os.environ.get("W7_SAE_DIR", "/root/sae"))
SEL_DIR = Path(os.environ.get("W7_SEL_DIR", str(SRC.parents[1] / "results/runs/SAE_20k_images_analysis")))
TORCH_SEED = 20260908

MODELS = {
  "supervised": dict(loader="hf", name="google/vit-base-patch16-224", sae="SAE_residual_APE_2_TOP64.pt",
                     sel="original_results/top_selective_features_per_position_residual.json", n_acc=5000),
  "augreg": dict(loader="timm", name="vit_base_patch16_224.augreg_in1k", sae="SAE_residual_APE_2_TOP64_third_seed.pt",
                 sel="third_seed/top_selective_features_per_position_residual_third_seed.json", n_acc=1000),
  "sam": dict(loader="timm", name="vit_base_patch16_224.sam_in1k", sae="SAE_residual_APE_2_TOP64_fourth_seed.pt",
              sel="fourth_seed/top_selective_features_per_position_residual_fourth_seed.json", n_acc=1000),
  "dino": dict(loader="timm", name="vit_base_patch16_224.dino", sae="SAE_residual_APE_2_TOP64_second_seed.pt",
               sel="second_seed/top_selective_features_per_position_residual_second_seed.json", n_acc=0),
  "deit3": dict(loader="timm", name="deit3_base_patch16_224.fb_in1k", sae=None, sel=None, n_acc=5000),
}
RUN = os.environ.get("W7_APE_MODELS", "supervised,augreg,sam,dino,deit3").split(",")
DATASET = load_imagenet(split="validation", streaming=True)

def save(tag, obj):
    (OUT / f"w7ape_{tag}_{STAMP}.json").write_text(json.dumps(obj, indent=2))

def load(m):
    cfg = MODELS[m]
    if cfg["loader"] == "hf":
        model, proc, src = load_ape(cfg["name"], device=DEV, half=False)
    else:
        model, proc, src = load_ape_timm(cfg["name"], device=DEV, half=False)
    return model.eval().float(), proc, src

def npt_of(model, src):
    return 1 if src == "transformers" else int(getattr(model, "num_prefix_tokens", 1))

def loader(proc, src, n):
    torch.manual_seed(TORCH_SEED)
    return prep_data(DATASET, proc, src, corruption_type=None, number_images=n, batch_size=BATCH, half=False, num_workers=0)

# ----------------------------------------------------------------- SAE edits
class SAEEdit:
    """Pre-hook on block 2: rewrite the residual through the SAE with an edit on latents."""
    def __init__(self, model, src, sae, mode, feats=None, err=False, mean=None, npt=1, seed=0):
        self.blk = get_vit_blocks(model, src)[2]; self.sae = sae; self.mode = mode
        self.f = torch.as_tensor(feats if feats is not None else [], dtype=torch.long, device=DEV)
        self.err = err; self.mean = mean; self.npt = npt; self.g = torch.Generator(device=DEV).manual_seed(seed); self.h = None
    def edit(self, z):
        if self.mode == "recon": return z
        f = self.f
        if self.mode == "zero": z[:, :, f] = 0.0
        elif self.mode == "mean": z[:, :, f] = self.mean[f].to(z.dtype)
        elif self.mode == "samepos":
            B = z.shape[0]; perm = torch.roll(torch.arange(B, device=z.device), 1)
            z[:, :, f] = z[perm][:, :, f]
        elif self.mode == "diffpos":
            T = z.shape[1]; p = self.npt
            perm = torch.cat([torch.arange(p, device=z.device), p + torch.randperm(T - p, generator=self.g, device=z.device)])
            z[:, :, f] = z[:, perm][:, :, f]
        return z
    def __enter__(self):
        def hook(mod, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            with torch.no_grad():
                z0 = self.sae._apply_sparsity(self.sae.preactivation(x))
                z1 = self.edit(z0.clone())
                x_new = self.sae.decode(z1)
                if self.err: x_new = x + (x_new - self.sae.decode(z0))
            return (x_new,) + tuple(inp[1:]) if isinstance(inp, tuple) else x_new
        self.h = self.blk.register_forward_pre_hook(hook); return self
    def __exit__(self, *e): self.h.remove()

class EmbedEdit:
    """No SAE: zero or permute the learned patch position embeddings (class-token slot kept)."""
    def __init__(self, model, src, mode, seed=0):
        self.p = model.vit.embeddings.position_embeddings if src == "transformers" else model.pos_embed
        self.mode = mode; self.seed = seed
    def __enter__(self):
        self.orig = self.p.data.clone(); d = self.p.data; T = d.shape[1]
        s = 1 if T in (197, 201) else 0  # class slot included in the table
        if self.mode == "zero": d[:, s:] = 0.0
        else:
            g = torch.Generator().manual_seed(self.seed); perm = torch.randperm(T - s, generator=g) + s
            d[:, s:] = self.orig[:, perm.to(d.device)]
        return self
    def __exit__(self, *e): self.p.data.copy_(self.orig)

class Null:
    def __enter__(self): return self
    def __exit__(self, *e): return False

# ----------------------------------------------------------------- readouts
def top1(model, proc, src, ctx_fn, n):
    corr = []
    with ctx_fn(), torch.inference_mode():
        for im, lab in loader(proc, src, n):
            im = im.to(DEV) if not isinstance(im, dict) else {k: v.to(DEV) for k, v in im.items()}
            logits = model(im) if src == "timm" else model(**im).logits
            corr.append((logits.argmax(-1).cpu() == torch.as_tensor(lab)).to(torch.uint8))
    c = torch.cat(corr); return float(c.float().mean()), c.numpy()

def ssdc(model, proc, src, ctx_fn):
    torch.manual_seed(TORCH_SEED); np.random.seed(TORCH_SEED % (2**31))
    with ctx_fn():
        s, _ = evaluate_ssdc(model, proc, DATASET, src, RPI=True, number_images=N_SS, batch_size=BATCH, metric="manhattan", half=False, num_workers=0)
    return [float(x) for x in s]

class CapIn:
    def __init__(s, model, src, blocks, npt):
        s.b = get_vit_blocks(model, src); s.which = blocks; s.npt = npt; s.src = src; s.store = {i: [] for i in blocks}; s.h = []
    def __enter__(s):
        for i in s.which:
            def hook(m, inp, out, i=i):
                t = inp[0].detach()[:, s.npt:, :]; s.store[i].append(t.half().cpu())
            s.h.append(get_block_attention(s.b[i], s.src).register_forward_hook(hook))
        return s
    def __exit__(s, *e):
        for x in s.h: x.remove()

def fit_probe_val(X, y, nc, n_img, T, seed=0):
    g = torch.Generator().manual_seed(seed); idx = torch.randperm(n_img, generator=g)
    ntr, nva = int(0.6 * n_img), int(0.2 * n_img)
    rows = lambda ids: (ids.unsqueeze(1) * T + torch.arange(T).unsqueeze(0)).reshape(-1)
    tr, va, te = rows(idx[:ntr]), rows(idx[ntr:ntr + nva]), rows(idx[ntr + nva:])
    Xg = X.to(DEV).float(); yg = y.to(DEV)
    mu = Xg[tr].mean(0, keepdim=True); sd = Xg[tr].std(0, keepdim=True).clamp_min(1e-6); Xg = (Xg - mu) / sd
    torch.manual_seed(seed); head = nn.Linear(Xg.shape[1], nc).to(DEV); opt = torch.optim.AdamW(head.parameters(), lr=5e-3)
    best_va, te_best, peak, last = -1, None, 0.0, None
    for ep in range(20):
        p = tr[torch.randperm(len(tr))]; head.train()
        for s0 in range(0, len(p), 512):
            b = p[s0:s0 + 512].to(DEV); opt.zero_grad(); F.cross_entropy(head(Xg[b]), yg[b]).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            av = torch.cat([(head(Xg[va[s:s+8192].to(DEV)]).argmax(-1) == yg[va[s:s+8192].to(DEV)]) for s in range(0, len(va), 8192)]).float().mean().item()
            at = torch.cat([(head(Xg[te[s:s+8192].to(DEV)]).argmax(-1) == yg[te[s:s+8192].to(DEV)]) for s in range(0, len(te), 8192)]).float().mean().item()
        peak = max(peak, at); last = at
        if av > best_va: best_va, te_best = av, at
    del Xg, head, opt; torch.cuda.empty_cache()
    return {"test_at_best_val": te_best, "final_test": last, "peak_test_old_protocol": peak, "best_val": best_va}

def probes(model, proc, src, ctx_fn, npt, H=14, W=14):
    cap = CapIn(model, src, PROBE_BLOCKS, npt)
    torch.manual_seed(TORCH_SEED)
    with ctx_fn(), cap:
        evaluate_ssdc(model, proc, DATASET, src, RPI=True, number_images=N_PROBE, batch_size=BATCH, metric="manhattan", half=False, num_workers=0)
    out = {}
    for b in PROBE_BLOCKS:
        X = torch.cat(cap.store[b], 0); n, T, D = X.shape; X = X.reshape(-1, D)
        pos = torch.arange(T).repeat(n)
        out[str(b)] = {nm: fit_probe_val(X, y, nc, n, T) for nm, y, nc in
                       [("exact", pos, T), ("row", pos // W, H), ("col", pos % W, W)]}
    return out

# ----------------------------------------------------------------- latent statistics
def latent_stats(model, proc, src, sae, npt):
    blk = get_vit_blocks(model, src)[2]; acc = {"n": 0, "fire": None, "mass": None, "sum": None}
    def hook(mod, inp):
        x = inp[0] if isinstance(inp, tuple) else inp
        with torch.no_grad():
            z = sae._apply_sparsity(sae.preactivation(x))[:, npt:, :].reshape(-1, sae.d_hidden).float()
            fire = (z > 0).float().sum(0); s = z.sum(0)
            acc["fire"] = fire if acc["fire"] is None else acc["fire"] + fire
            acc["sum"] = s if acc["sum"] is None else acc["sum"] + s
            acc["n"] += z.shape[0]
    h = blk.register_forward_pre_hook(hook)
    try:
        torch.manual_seed(TORCH_SEED)
        evaluate_ssdc(model, proc, DATASET, src, RPI=True, number_images=N_STATS, batch_size=BATCH, metric="manhattan", half=False, num_workers=0)
    finally: h.remove()
    fire = acc["fire"] / acc["n"]; meanz = acc["sum"] / acc["n"]
    mass = meanz * sae.W_dec.detach().norm(dim=1)
    return fire.cpu(), meanz.cpu(), mass.cpu()

def matched_draws(pos, fire, mass, n_draws=5, k_near=10, seed=0):
    rng = np.random.default_rng(seed); pos_set = set(pos)
    cand = np.array([i for i in range(len(fire)) if i not in pos_set])
    feat = np.stack([np.log(fire.numpy() + 1e-6), np.log(np.abs(mass.numpy()) + 1e-8)], 1)
    feat = (feat - feat[cand].mean(0)) / (feat[cand].std(0) + 1e-8)
    draws = []
    for _ in range(n_draws):
        used = set(); d = []
        for p in pos:
            dist = np.linalg.norm(feat[cand] - feat[p], axis=1); order = cand[np.argsort(dist)]
            pool = [c for c in order[:k_near * 3] if c not in used][:k_near]
            c = int(rng.choice(pool)); used.add(c); d.append(c)
        draws.append(d)
    return draws

# ----------------------------------------------------------------- main loop
def run_model(m):
    cfg = MODELS[m]; t0 = time.time(); res = {"model": m, "checkpoint": cfg["name"], "conditions": {}}
    model, proc, src = load(m); npt = npt_of(model, src); res["num_prefix_tokens"] = npt
    conds = [("intact", lambda: Null()), ("embed_zero", lambda: EmbedEdit(model, src, "zero")),
             ("embed_perm", lambda: EmbedEdit(model, src, "perm"))]
    probe_set = {"intact", "embed_zero", "embed_perm"}
    if cfg["sae"]:
        sd = torch.load(SAE_DIR / cfg["sae"], map_location="cpu")
        sae = SAE(768, sd["W_dec"].shape[0], architecture="topk", k=64).to(DEV); sae.load_state_dict(sd); sae.eval()
        pos = load_top_features(str(SEL_DIR / cfg["sel"]), "APE", 2, axis="both"); res["positional_features"] = pos
        fire, meanz, mass = latent_stats(model, proc, src, sae, npt)
        res["latent_stats"] = {"fire_pos": fire[pos].tolist(), "mass_pos": mass[pos].tolist(),
                               "fire_all_median": float(fire.median()), "fire_all_mean": float(fire.mean())}
        g = torch.Generator().manual_seed(0); uni = torch.randperm(sae.d_hidden, generator=g)[:len(pos)].tolist()
        draws = matched_draws(pos, fire, mass); res["uniform_draw"] = uni; res["matched_draws"] = draws
        res["latent_stats"]["fire_uniform"] = fire[uni].tolist(); res["latent_stats"]["fire_matched0"] = fire[draws[0]].tolist()
        E = lambda mode, feats=None, err=False, seed=0: (lambda: SAEEdit(model, src, sae, mode, feats, err, meanz.to(DEV), npt, seed))
        conds += [("recon", E("recon")), ("pos_zero", E("zero", pos)), ("rand_uniform", E("zero", uni)),
                  ("recon_err", E("recon", None, True)), ("pos_zero_err", E("zero", pos, True)),
                  ("pos_mean_err", E("mean", pos, True)),
                  ("pos_resample_samepos_err", E("samepos", pos, True)),
                  ("pos_resample_diffpos_err", E("diffpos", pos, True))]
        for i, d in enumerate(draws):
            conds.append((f"rand_matched_{i}", E("zero", d)))
        conds.append(("rand_matched_0_err", E("zero", draws[0], True)))
        probe_set |= {"recon", "pos_zero", "rand_uniform", "rand_matched_0", "pos_zero_err", "pos_resample_diffpos_err", "pos_resample_samepos_err"}
    has_head = cfg["n_acc"] > 0; perimg = {}
    for name, cf in conds:
        t = time.time(); rec = {}
        try:
            rec["ssdc"] = ssdc(model, proc, src, cf)
            if has_head:
                rec["acc"], perimg[name] = top1(model, proc, src, cf, cfg["n_acc"])
            if name in probe_set:
                rec["probes"] = probes(model, proc, src, cf, npt)
        except Exception as ex:
            rec["error"] = repr(ex); traceback.print_exc()
        rec["runtime_s"] = round(time.time() - t, 1); res["conditions"][name] = rec
        print(f"  {m} {name:26s} ssdc2={rec.get('ssdc',[None]*12)[2]} acc={rec.get('acc')} {rec['runtime_s']}s", flush=True)
        save(m, res)
    if perimg: np.savez_compressed(PER / f"ape_{m}_correct.npz", **perimg)
    res["runtime_s"] = round(time.time() - t0, 1); save(m, res)
    del model; torch.cuda.empty_cache()

for m in RUN:
    print(f"\n=== APE {m} ===", flush=True)
    try: run_model(m)
    except Exception as ex: print("FAILED", m, repr(ex)); traceback.print_exc()
print("W7 APE DONE", flush=True)
