import subprocess, sys

# ---- cell 2 ----
import importlib, subprocess, sys

def ensure(pkg, import_name=None):
    try:
        importlib.import_module(import_name or pkg)
        return False
    except ImportError:
        print(f"installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        return True

for p in ["timm", "transformers", "datasets", "scipy"]:
    ensure(p)
print("environment ready")

# ---- cell 3 ----
import os, json, math, time, warnings, itertools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

warnings.filterwarnings("ignore", category=UserWarning)

print("torch :", torch.__version__)
print("timm  :", timm.__version__)
print("cuda  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

# ---- cell 5 ----
# ============================================================================
# CONFIG
# ============================================================================
REPO_URL      = None
REPO_PATH     = str(Path(__file__).resolve().parents[2])

NUMBER_IMAGES = 1000
BATCH_SIZE    = 256
METRIC        = "manhattan"
TORCH_SEED    = 20260908    # reseeded before EVERY pass, as in W1-B
MASK_SEED     = 771         # fixed seed for the random-plane control subsets

INPUT_SIZE    = 224         # forced for all four models, as in the reference runs

W1B_JSON      = None        # w1b_floors_*.json; None = auto-search
CALIB_TOL     = 0.05

# --- probe settings, following LINEAR_PROBE.md ------------------------------
PROBE_LR         = 5e-3
PROBE_PASSES     = 20
PROBE_BATCH      = 512
PROBE_TRAIN_FRAC = 0.8      # split by IMAGE, never by token
PROBE_BLOCKS     = "sparse" # "sparse" | "all" | "none"

OUT_DIR = Path("w2_output")
OUT_DIR.mkdir(exist_ok=True)
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Loaders are named explicitly. load_model() is NOT used anywhere: load_model('dinov3')
# omits input_size, which silently yields a 16x16 grid; every reference run went
# through load_rope(input_size=224).
MODELS = {
    "APE · google ViT": dict(loader="load_ape",      kwargs={},
                             family="APE",  probe_layer=2),
    "RoPE · naver":     dict(loader="load_rope",     kwargs=dict(input_size=INPUT_SIZE),
                             family="RoPE", probe_layer=4),
    "APE · DINOv1":     dict(loader="load_ape_timm", kwargs=dict(input_size=INPUT_SIZE),
                             family="APE",  probe_layer=2),
    "RoPE · DINOv3":    dict(loader="load_rope",
                             kwargs=dict(model_name="vit_base_patch16_dinov3.lvd1689m",
                                         input_size=INPUT_SIZE),
                             family="RoPE", probe_layer=4),
}

DOSES = [4, 8, 16, 24]      # plane counts; 32 (= all) is the floor condition
# ============================================================================

GATES = []
def gate(name, scope, passed, detail=""):
    GATES.append({"gate": name, "scope": scope,
                  "status": "PASS" if passed else "FAIL", "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {scope}: {name}" + (f": {detail}" if detail else ""))
    return passed

print(f"device {DEVICE} | dtype {DTYPE} | fp32 (no half path) | images {NUMBER_IMAGES} "
      f"| batch {BATCH_SIZE}")
print(f"run {RUN_STAMP}")

# ---- cell 7 ----
def resolve_repo_path():
    '''Find a checkout with metrics/ and main/ as direct children, cloning if needed.'''
    cands = []
    if REPO_PATH:
        cands.append(Path(REPO_PATH).expanduser().resolve())
    clone = Path("vit-sae-analysis")
    cands += [clone / "project_code" / "src", clone / "src", clone]
    for c in cands:
        if (c / "metrics").exists() and (c / "main").exists():
            return c
    if REPO_URL and not clone.exists():
        print(f"cloning {REPO_URL} ...")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(clone)])
        for c in cands:
            if (c / "metrics").exists() and (c / "main").exists():
                return c
    raise FileNotFoundError(
        "No checkout with metrics/ and main/ found. Tried: "
        + ", ".join(str(c) for c in cands)
        + ". If the repo is private, clone manually and set REPO_PATH.")

repo = resolve_repo_path()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
print("repo root:", repo)

from metrics.ssdc import evaluate_ssdc, spatial_similarity_distance_correlation
from main.load_models import (get_vit_blocks, get_block_attention,
                              get_patch_embed_conv,
                              load_ape, load_rope, load_ape_timm)
from main.model import predict
from main.prep_data import prep_data

LOADERS = {"load_ape": load_ape, "load_rope": load_rope, "load_ape_timm": load_ape_timm}
print("imported evaluate_ssdc, load_ape/load_rope/load_ape_timm, predict, prep_data")

from scipy.spatial.distance import cdist as _cdist
_G = 14
_c = np.stack(np.meshgrid(np.arange(_G), np.arange(_G), indexing="ij"), -1).reshape(-1, 2)
_D = _cdist(_c, _c, metric="cityblock")
_a = spatial_similarity_distance_correlation(-_D, grid_size=_G, metric="manhattan")
_b = spatial_similarity_distance_correlation(_D, grid_size=_G, metric="manhattan")
_r = spatial_similarity_distance_correlation(
    np.random.default_rng(0).normal(size=(_G*_G, _G*_G)), grid_size=_G, metric="manhattan")
gate("imported metric satisfies its unit test", "repo-metric",
     _a > 0.999 and _b < -0.999 and abs(_r) < 0.1, f"{_a:+.3f} / {_b:+.3f} / {_r:+.3f}")

# ---- cell 8 ----
def load_val_dataset():
    from experiments.common import load_imagenet
    return load_imagenet(split="validation", streaming=True)

DATASET = load_val_dataset()
print("dataset ready:", type(DATASET).__name__)

# ---- cell 10 ----
def find_rope_module(model):
    for name, mod in model.named_modules():
        if callable(getattr(mod, "get_embed", None)):
            return name, mod
    return None, None


def capture_forward_grid(model, processor, source):
    '''Run one tiny batch and record the (H, W) the model actually produces.'''
    grid = {}
    conv = get_patch_embed_conv(model, source)
    h = conv.register_forward_hook(
        lambda m, i, o: grid.update(hw=(int(o.shape[2]), int(o.shape[3]))
                                    if o.dim() == 4 else None))
    dl = prep_data(DATASET, processor, source, corruption_type=None,
                   number_images=BATCH_SIZE, batch_size=BATCH_SIZE,
                   half=False, num_workers=0)
    with torch.no_grad():
        for images, _ in dl:
            if source == "transformers":
                images = {k: v.to(DEVICE) for k, v in images.items()}
                model(**images)
            else:
                model(images.to(DEVICE))
            break
    h.remove()
    return grid.get("hw")


def read_rotate_half(model, source):
    '''timm stores the layout flag on the attention module (eva.py line ~736).'''
    for blk in get_vit_blocks(model, source):
        attn = get_block_attention(blk, source)
        for attr in ("rotate_half", "half", "rope_rotate_half"):
            v = getattr(attn, attr, None)
            if isinstance(v, bool):
                return v, attr
    return False, "default(interleaved)"


def plane_columns(p, n_planes, rotate_half):
    '''Column indices inside ONE half (sin or cos) that carry rotation plane p.'''
    return [p, p + n_planes] if rotate_half else [2 * p, 2 * p + 1]

# ---- cell 11 ----
def derive_axis_map(emb, n_planes, rotate_half, H, W):
    '''Classify every rotation plane as row-carrying or column-carrying, empirically.

    For plane p take z = cos + i*sin at one of its columns and reshape to (H, W).
    A plane that encodes only the row index is constant ALONG a row, so its residual
    after subtracting the row-mean is ~0. Using the complex exponential rather than the
    angle makes this immune to phase wrapping at high frequencies.
    '''
    sin, cos = emb.detach().float().chunk(2, -1)
    rows, decisive = [], True
    for p in range(n_planes):
        c = plane_columns(p, n_planes, rotate_half)[0]
        z = torch.complex(cos[:, c], sin[:, c]).reshape(H, W)
        resid_along_cols = (z - z.mean(dim=1, keepdim=True)).abs().mean().item()
        resid_along_rows = (z - z.mean(dim=0, keepdim=True)).abs().mean().item()
        axis = "row" if resid_along_cols < resid_along_rows else "col"
        lo, hi = sorted([resid_along_cols, resid_along_rows])
        ratio = lo / hi if hi > 1e-12 else 1.0
        if ratio > 0.10:
            decisive = False
        rows.append({"plane": p, "axis": axis, "resid_along_cols": resid_along_cols,
                     "resid_along_rows": resid_along_rows, "ratio": ratio})
    row_planes = [r["plane"] for r in rows if r["axis"] == "row"]
    col_planes = [r["plane"] for r in rows if r["axis"] == "col"]
    return rows, row_planes, col_planes, decisive

# ---- cell 13 ----
def mask_planes(emb, kill, n_planes, rotate_half):
    '''Set the listed rotation planes to identity (sin=0, cos=1). kill=[] is a no-op.'''
    if not kill:
        return emb
    sin, cos = emb.chunk(2, -1)
    sin, cos = sin.clone(), cos.clone()
    for p in kill:
        for c in plane_columns(p, n_planes, rotate_half):
            sin[..., c] = 0.0
            cos[..., c] = 1.0
    return torch.cat([sin, cos], -1)


def identity_like(emb):
    sin, cos = emb.chunk(2, -1)
    return torch.cat([torch.zeros_like(sin), torch.ones_like(cos)], -1)


class RopeGlobalOverride:
    '''Wrap rope.get_embed; affects every block.'''
    def __init__(self, model, transform):
        self.name, self.mod = find_rope_module(model)
        self.transform = transform
        self.orig = None

    def __enter__(self):
        if self.mod is None:
            raise RuntimeError("no module exposing get_embed(): not a RoPE checkpoint?")
        self.orig = self.mod.get_embed
        o, tf = self.orig, self.transform
        self.mod.get_embed = (lambda _o=o, _tf=tf: (lambda *a, **k: _tf(_o(*a, **k))))()
        return self

    def __exit__(self, *exc):
        self.mod.get_embed = self.orig


class RopePerBlockOverride:
    '''Rewrite the rope kwarg on selected blocks only. Used by W3; gated here.'''
    def __init__(self, model, source, transform, blocks=None):
        self.blocks = get_vit_blocks(model, source)
        self.which = list(range(len(self.blocks))) if blocks is None else list(blocks)
        self.transform = transform
        self.handles = []

    def __enter__(self):
        tf = self.transform
        def make(idx):
            def pre(module, args, kwargs):
                r = kwargs.get("rope", None)
                if r is not None:
                    kwargs = dict(kwargs)
                    kwargs["rope"] = tf(r)
                return args, kwargs
            return pre
        for i in self.which:
            attn = get_block_attention(self.blocks[i], "timm")
            self.handles.append(attn.register_forward_pre_hook(make(i), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


class APEZero:
    '''Zero the positional embedding. The APE analogue of the rotary floor.'''
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        cands = [n for n, _ in self.model.named_parameters()
                 if n.endswith("pos_embed") or n.endswith("position_embeddings")]
        if not cands:
            raise RuntimeError("no positional-embedding parameter found")
        self.name = sorted(cands, key=len)[0]
        parts = self.name.split(".")
        mod = self.model
        for p in parts[:-1]:
            mod = getattr(mod, p)
        self.parent, self.attr = mod, parts[-1]
        pe = getattr(self.parent, self.attr)
        self.saved = pe.detach().clone()
        with torch.no_grad():
            pe.zero_()
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            getattr(self.parent, self.attr).copy_(self.saved)


class NullCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False

# ---- cell 15 ----
def _rot_interleaved(x):
    x = x.unflatten(-1, (-1, 2))
    return torch.stack([-x[..., 1], x[..., 0]], -1).flatten(-2)

def _apply_interleaved(x, emb):
    s, c = emb.chunk(2, -1)
    return x * c + _rot_interleaved(x) * s

for layout_name, rh in [("interleaved (naver)", False), ("rotate_half (DINOv3)", True)]:
    npl, d = 32, 64
    e = torch.randn(196, 2 * d)
    gate(f"empty mask is bit-identical [{layout_name}]", "rope-algebra",
         torch.equal(mask_planes(e, [], npl, rh), e))
    gate(f"full mask equals identity [{layout_name}]", "rope-algebra",
         torch.allclose(mask_planes(e, list(range(npl)), npl, rh), identity_like(e)))
    # masking plane p must touch exactly the columns plane_columns(p) says it does
    m = mask_planes(e, [3], npl, rh)
    diff_cols = torch.nonzero((m - e).abs().sum(0) > 0).flatten().tolist()
    sin_cols = plane_columns(3, npl, rh)
    expected = sorted(set(sin_cols + [c + d for c in sin_cols]))
    gate(f"masking plane 3 touches only its own columns [{layout_name}]", "rope-algebra",
         diff_cols == expected, f"touched {diff_cols}, expected {expected}")

x = torch.randn(2, 196, 64)
e = torch.randn(196, 128)
gate("identity embedding is a no-op on tokens", "rope-algebra",
     torch.allclose(_apply_interleaved(x, identity_like(e)), x, atol=1e-6))
gate("a real embedding does change tokens", "rope-algebra",
     not torch.allclose(_apply_interleaved(x, e), x))

# ---- cell 17 ----
class Capture:
    '''Collects the block-attn input (post-norm residual) at chosen blocks, prefix stripped.'''
    def __init__(self, model, source, blocks, n_prefix):
        self.blocks = get_vit_blocks(model, source)
        self.which = list(blocks)
        self.n_prefix = int(n_prefix)
        self.source = source
        self.store = {i: [] for i in self.which}
        self.handles = []

    def stacked(self, i):
        return torch.cat(self.store[i], 0) if self.store[i] else None


def make_capture(model, source, blocks, n_prefix):
    return Capture(model, source, blocks, n_prefix)


def _cap_enter(cap):
    def make(i):
        def hook(module, inputs, output):
            t = inputs[0].detach()
            if cap.n_prefix > 0:
                t = t[:, cap.n_prefix:, :]
            cap.store[i].append(t.float().cpu())
        return hook
    for i in cap.which:
        attn = get_block_attention(cap.blocks[i], cap.source)
        cap.handles.append(attn.register_forward_hook(make(i)))
    return cap


def _cap_exit(cap):
    for h in cap.handles:
        h.remove()
    cap.handles = []

# ---- cell 18 ----
def train_linear_probe(feats, labels, n_classes, n_images, tokens_per_image,
                       lr=PROBE_LR, passes=PROBE_PASSES, batch=PROBE_BATCH,
                       train_frac=PROBE_TRAIN_FRAC, seed=TORCH_SEED):
    '''feats (N_img*T, D) cpu float32; labels (T,) tiled per image. Split by image.'''
    g = torch.Generator().manual_seed(seed)
    n_train_img = int(round(n_images * train_frac))
    idx = torch.randperm(n_images, generator=g)
    tr_img, te_img = idx[:n_train_img], idx[n_train_img:]

    def rows_for(img_ids):
        base = (img_ids.unsqueeze(1) * tokens_per_image)
        return (base + torch.arange(tokens_per_image).unsqueeze(0)).reshape(-1)

    tr, te = rows_for(tr_img), rows_for(te_img)
    y_all = labels.repeat(n_images)

    Xtr, ytr = feats[tr].to(DEVICE), y_all[tr].to(DEVICE)
    Xte, yte = feats[te].to(DEVICE), y_all[te].to(DEVICE)

    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    head = nn.Linear(Xtr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    n = Xtr.shape[0]
    peak = 0.0
    for _ in range(passes):
        perm = torch.randperm(n, device=DEVICE)
        head.train()
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            loss = F.cross_entropy(head(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            acc = 0.0
            for s in range(0, Xte.shape[0], 4096):
                acc += (head(Xte[s:s+4096]).argmax(-1) == yte[s:s+4096]).sum().item()
            acc /= Xte.shape[0]
        peak = max(peak, acc)          # peak accuracy, as in LINEAR_PROBE.md
    del Xtr, Xte, ytr, yte, head
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return float(peak)


def probe_all_heads(feats, H, W, n_images):
    T = H * W
    pos = torch.arange(T)
    out = {}
    out["exact"] = train_linear_probe(feats, pos, T, n_images, T)
    out["row"] = train_linear_probe(feats, pos // W, H, n_images, T)
    out["col"] = train_linear_probe(feats, pos % W, W, n_images, T)
    out["chance"] = {"exact": 1.0 / T, "row": 1.0 / H, "col": 1.0 / W}
    return out

# ---- cell 20 ----
def measure(model, processor, source, ctx, probe_blocks, n_prefix, H, W, tag, probe_layer):
    '''One condition: SSDC at all blocks, plus probes at probe_blocks.'''
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED % (2**31))
    t0 = time.time()

    cap = make_capture(model, source, probe_blocks, n_prefix) if probe_blocks else None
    if cap:
        _cap_enter(cap)
    try:
        with ctx:
            scores, _ = evaluate_ssdc(
                model, processor, DATASET, source,
                RPI=True, number_images=NUMBER_IMAGES, batch_size=BATCH_SIZE,
                metric=METRIC, half=False, num_workers=0,
            )
    finally:
        if cap:
            _cap_exit(cap)

    res = {"ssdc": [float(s) for s in scores], "probes": {}}
    if cap:
        for i in probe_blocks:
            acts = cap.stacked(i)
            if acts is None:
                continue
            n_img = acts.shape[0]
            feats = acts.reshape(-1, acts.shape[-1])
            res["probes"][str(i)] = probe_all_heads(feats, H, W, n_img)
            del acts, feats
        cap.store.clear()
    res["runtime_s"] = round(time.time() - t0, 1)
    print(f"    {tag}: ssdc@probe={res['ssdc'][probe_layer]:+.4f}  {res['runtime_s']}s")
    return res


def accuracy(model, processor, source):
    accs = {}
    for tag, rpi in (("clean", False), ("rpi", True)):
        torch.manual_seed(TORCH_SEED)
        dl = prep_data(DATASET, processor, source, corruption_type=None,
                       number_images=NUMBER_IMAGES, batch_size=BATCH_SIZE,
                       half=False, num_workers=0)
        accs[tag] = float(predict(model, dl, source, RPI=rpi, half=False))
    return accs

# ---- cell 22 ----
def load_w1b():
    if W1B_JSON and Path(W1B_JSON).exists():
        return json.load(open(W1B_JSON))
    for pat in ["w1b_floors_*.json", "**/w1b_floors_*.json"]:
        hits = sorted(Path(".").glob(pat))
        if hits:
            print("W1-B reference:", hits[-1])
            return json.load(open(hits[-1]))
    print("W1-B reference not found: calibration will be SKIPPED")
    return None

W1B = load_w1b()
REF = {}
if W1B:
    for lab, e in W1B.get("results", {}).items():
        if "supplementary" in lab or e.get("status") != "ok":
            continue
        L = e["probe_layer"]
        REF[lab] = {"baseline": e["baseline_ssdc"][L], "floor": e["floor_ssdc"][L],
                    "prefix": e.get("num_prefix_tokens")}
    print("\nW1-B fp16 reference at probe layer:")
    for k, v in REF.items():
        print(f"  {k:20s} baseline {v['baseline']:+.4f}  floor {v['floor']:+.4f}")

# ---- cell 24 ----
def build_conditions(family, n_planes, row_planes, col_planes, axis_rows):
    conds = [("baseline", None), ("floor", "ALL")]
    if family != "RoPE":
        return conds
    order = sorted(range(n_planes),
                   key=lambda p: axis_rows[p]["freq_rank"], reverse=True)  # high freq first
    rng = np.random.default_rng(MASK_SEED)
    for k in DOSES:
        per_axis = k // 2
        hi = ([p for p in order if p in row_planes][:per_axis]
              + [p for p in order if p in col_planes][:per_axis])
        if len(hi) < k:
            hi = order[:k]
        conds.append((f"dose_hi_{k}", sorted(hi)))
    for k in DOSES:
        conds.append((f"random_{k}", sorted(rng.choice(n_planes, size=k, replace=False).tolist())))
    conds.append(("axis_row", sorted(row_planes)))
    conds.append(("axis_col", sorted(col_planes)))
    return conds


def freq_rank_table(emb, n_planes, rotate_half, H, W):
    '''Rank planes by spatial frequency: how fast the phase turns across the grid.'''
    sin, cos = emb.detach().float().chunk(2, -1)
    out = {}
    for p in range(n_planes):
        c = plane_columns(p, n_planes, rotate_half)[0]
        z = torch.complex(cos[:, c], sin[:, c]).reshape(H, W)
        dr = (z[1:, :] - z[:-1, :]).abs().mean().item()
        dc = (z[:, 1:] - z[:, :-1]).abs().mean().item()
        out[p] = {"freq_rank": max(dr, dc)}
    return out

# ---- cell 25 ----
RESULTS = {}

for label, cfg in MODELS.items():
    print("\n" + "=" * 78)
    print(f"{label}   ({cfg['loader']}{cfg['kwargs']})")
    print("=" * 78)
    t_model = time.time()
    PROBE_L = cfg["probe_layer"]

    try:
        model, processor, source = LOADERS[cfg["loader"]](
            device=DEVICE, half=False, **cfg["kwargs"])
        model.eval().float()
    except Exception as e:
        gate("model loads", label, False, f"{type(e).__name__}: {e}")
        RESULTS[label] = {"status": "load_failed", "error": str(e)}
        continue

    n_blocks = len(get_vit_blocks(model, source))
    n_prefix = 1 if source == "transformers" else int(getattr(model, "num_prefix_tokens", 1))
    hw = capture_forward_grid(model, processor, source)
    if hw is None:
        raise RuntimeError("could not read the forward-pass grid from the patch-embed conv")
    H, W = hw
    T = H * W
    print(f"  source={source} blocks={n_blocks} prefix={n_prefix} grid={H}x{W} tokens={T}")

    entry = {"status": "ok", "loader": cfg["loader"], "loader_kwargs": {k: str(v) for k, v in cfg["kwargs"].items()},
             "source": source, "family": cfg["family"], "probe_layer": PROBE_L,
             "n_blocks": n_blocks, "num_prefix_tokens": n_prefix,
             "grid": [H, W], "patch_tokens": T, "dtype": "float32"}

    gate("forward grid is 14x14 (196 tokens)", label, (H, W) == (14, 14), f"{H}x{W}")
    if hasattr(model, "patch_embed"):
        gs = tuple(getattr(model.patch_embed, "grid_size", ()))
        entry["patch_embed_grid_size"] = list(gs)
        if gs and gs != (H, W):
            print(f"    note: patch_embed.grid_size says {gs} but the forward pass is "
                  f"{(H, W)}: using the forward pass, as intended")

    if PROBE_BLOCKS == "all":
        pblocks = list(range(n_blocks))
    elif PROBE_BLOCKS == "none":
        pblocks = []
    else:
        pblocks = sorted({PROBE_L, min(PROBE_L + 2, n_blocks - 1),
                          min(PROBE_L + 4, n_blocks - 1), n_blocks - 1})
    entry["probe_blocks"] = pblocks
    print(f"  probe blocks: {pblocks}")

    # ---- rotary introspection -------------------------------------------
    row_planes = col_planes = []
    n_planes = None
    if cfg["family"] == "RoPE":
        rname, rmod = find_rope_module(model)
        rotate_half, rh_src = read_rotate_half(model, source)
        emb0 = rmod.get_embed(shape=(H, W))
        n_planes = int(emb0.shape[-1] // 4)
        entry.update({"rope_module": rname, "rotate_half": bool(rotate_half),
                      "rotate_half_source": rh_src, "n_planes": n_planes,
                      "rope_rows": int(emb0.shape[0])})
        print(f"  rope='{rname}' planes={n_planes} rotate_half={rotate_half} ({rh_src}) "
              f"rope_rows={emb0.shape[0]}")

        gate("rope tensor row count equals patch-token count", label,
             int(emb0.shape[0]) == T, f"rope {emb0.shape[0]} vs tokens {T}")

        amap, row_planes, col_planes, decisive = derive_axis_map(emb0, n_planes, rotate_half, H, W)
        entry["axis_map"] = amap
        entry["row_planes"], entry["col_planes"] = row_planes, col_planes
        print(f"  axis split: {len(row_planes)} row / {len(col_planes)} col")
        gate("every plane classifies decisively as row or col", label, decisive,
             f"max ratio {max(a['ratio'] for a in amap):.3f}")
        gate("axis split is balanced", label,
             len(row_planes) == len(col_planes) == n_planes // 2,
             f"{len(row_planes)}/{len(col_planes)}")

        # global vs per-block equivalence (used by W3)
        try:
            torch.manual_seed(TORCH_SEED)
            with RopeGlobalOverride(model, identity_like):
                g_emb = rmod.get_embed(shape=(H, W)).detach().float().clone()
            same = torch.allclose(g_emb, identity_like(emb0).float(), atol=1e-6)
            gate("global override yields the identity embedding", label, same)
        except Exception as e:
            gate("global override yields the identity embedding", label, False, str(e))

        entry["freq_rank"] = {str(k): v["freq_rank"]
                              for k, v in freq_rank_table(emb0, n_planes, rotate_half, H, W).items()}
        axis_rows = {a["plane"]: {"freq_rank": entry["freq_rank"][str(a["plane"])]} for a in amap}
    else:
        axis_rows = {}

    # ---- condition sweep -------------------------------------------------
    conds = build_conditions(cfg["family"], n_planes, row_planes, col_planes, axis_rows)
    entry["conditions"] = {}
    print(f"  {len(conds)} conditions")

    for cname, kill in conds:
        if cfg["family"] == "RoPE":
            if kill is None:
                ctx = NullCtx()
            elif kill == "ALL":
                ctx = RopeGlobalOverride(model, identity_like)
            else:
                ctx = RopeGlobalOverride(
                    model, lambda e, k=kill: mask_planes(e, k, n_planes, rotate_half))
        else:
            ctx = NullCtx() if kill is None else APEZero(model)

        try:
            r = measure(model, processor, source, ctx, pblocks, n_prefix, H, W, cname, PROBE_L)
            r["planes_killed"] = (kill if isinstance(kill, list)
                                  else ([] if kill is None else list(range(n_planes or 0))))
            entry["conditions"][cname] = r
        except Exception as e:
            print(f"    {cname}: FAILED {type(e).__name__}: {e}")
            entry["conditions"][cname] = {"status": "failed", "error": str(e)}

    # ---- live gates on the measured numbers ------------------------------
    b = entry["conditions"].get("baseline", {}).get("ssdc")
    f = entry["conditions"].get("floor", {}).get("ssdc")
    if b and f:
        entry["baseline_ssdc"], entry["floor_ssdc"] = b, f
        entry["achievable_range"] = b[PROBE_L] - f[PROBE_L]
        gate("floor is below baseline", label, f[PROBE_L] < b[PROBE_L] - 0.01,
             f"{b[PROBE_L]:+.4f} -> {f[PROBE_L]:+.4f}")
        if cfg["family"] == "RoPE":
            gate("floor equals baseline at block 0 (rope acts inside attention)", label,
                 abs(b[0] - f[0]) < 1e-9, f"delta {abs(b[0]-f[0]):.2e}")
        ref = REF.get(label)
        if ref:
            d = abs(b[PROBE_L] - ref["baseline"])
            gate("fp32 baseline matches W1-B fp16", label, d <= CALIB_TOL,
                 f"{b[PROBE_L]:+.4f} vs {ref['baseline']:+.4f} (delta {d:.4f})")
            entry["w1b_reference"] = ref
            entry["fp32_vs_fp16_baseline_delta"] = float(d)
            entry["fp32_vs_fp16_floor_delta"] = float(abs(f[PROBE_L] - ref["floor"]))

    if cfg["family"] == "RoPE" and "axis_row" in entry["conditions"]:
        try:
            ar = entry["conditions"]["axis_row"]["ssdc"][PROBE_L]
            ac = entry["conditions"]["axis_col"]["ssdc"][PROBE_L]
            r16 = entry["conditions"].get("random_16", {}).get("ssdc", [None]*12)[PROBE_L]
            print(f"  axis_row {ar:+.4f} | axis_col {ac:+.4f} | random_16 {r16}")
        except Exception:
            pass

    try:
        entry["baseline_acc"] = accuracy(model, processor, source)
    except Exception as e:
        entry["baseline_acc"] = {"error": str(e)}

    entry["runtime_s"] = round(time.time() - t_model, 1)
    RESULTS[label] = entry
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"  model done in {entry['runtime_s']}s")

# ---- cell 27 ----
import pandas as pd
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

rows = []
for label, e in RESULTS.items():
    if e.get("status") != "ok" or "baseline_ssdc" not in e:
        continue
    L, b, f = e["probe_layer"], e["baseline_ssdc"], e["floor_ssdc"]
    rng = b[L] - f[L]
    for cname, r in e["conditions"].items():
        if "ssdc" not in r:
            continue
        v = r["ssdc"][L]
        pr = r.get("probes", {}).get(str(L), {})
        rows.append({
            "run": label, "condition": cname,
            "n_killed": len(r.get("planes_killed", [])),
            "ssdc": v,
            "retention": (v - f[L]) / rng if rng > 0 else np.nan,
            "damage": 1 - (v - f[L]) / rng if rng > 0 else np.nan,
            "probe_exact": pr.get("exact"), "probe_row": pr.get("row"), "probe_col": pr.get("col"),
        })
summary = pd.DataFrame(rows)
print("W2 SUMMARY: at each model's probe layer\n")
for label in RESULTS:
    sub = summary[summary.run == label]
    if sub.empty:
        continue
    print(f"--- {label} " + "-" * (58 - len(label)))
    print(sub.drop(columns=["run"]).to_string(index=False))
    print()

# ---- cell 28 ----
# axis selectivity
print("AXIS SELECTIVITY (probe accuracy at the probe layer)\n")
for label, e in RESULTS.items():
    if e.get("family") != "RoPE" or e.get("status") != "ok":
        continue
    L = str(e["probe_layer"])
    cs = e["conditions"]
    def pr(c, h):
        return cs.get(c, {}).get("probes", {}).get(L, {}).get(h)
    print(f"--- {label}")
    print(f"{'condition':12s} {'exact':>8} {'row':>8} {'col':>8}")
    for c in ["baseline", "random_16", "axis_row", "axis_col", "floor"]:
        e_, r_, c_ = pr(c, "exact"), pr(c, "row"), pr(c, "col")
        fmt = lambda x: f"{x:8.3f}" if isinstance(x, float) else f"{'--':>8}"
        print(f"{c:12s} {fmt(e_)} {fmt(r_)} {fmt(c_)}")
    print("  expectation: killing row planes should hurt the row head far more than the col head")
    print()

# ---- cell 29 ----
n_ok = [(l, e) for l, e in RESULTS.items()
        if e.get("status") == "ok" and "baseline_ssdc" in e]
if n_ok:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(n_ok), figsize=(4.4 * len(n_ok), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, e) in zip(axes, n_ok):
        L, b, f = e["probe_layer"], e["baseline_ssdc"], e["floor_ssdc"]
        rng = b[L] - f[L]
        x = range(e["n_blocks"])
        ax.plot(x, b, "o-", lw=2, ms=4, color="black", label="baseline")
        ax.plot(x, f, "s--", lw=1.6, ms=4, color="crimson", label="floor")
        cm = plt.get_cmap("viridis")
        doses = [c for c in e["conditions"] if c.startswith("dose_hi_")]
        for i, c in enumerate(sorted(doses, key=lambda s: int(s.split("_")[-1]))):
            r = e["conditions"][c]
            if "ssdc" in r:
                ax.plot(x, r["ssdc"], "-", lw=1.4,
                        color=cm(i / max(len(doses) - 1, 1)), label=c)
        ax.axvline(L, color="k", ls=":", lw=1.1, alpha=0.6)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("block")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("SSDC under RPI")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Dose-response: rotation planes removed globally", y=1.02)
    fig.tight_layout()
    plt.show()

# ---- cell 31 ----
gdf = pd.DataFrame(GATES)
print("GATE REPORT\n")
print(gdf.to_string(index=False) if len(gdf) else "(none)")
n_fail = int((gdf.status == "FAIL").sum()) if len(gdf) else 0
print("\n" + "=" * 78)
print("ALL GATES PASSED" if n_fail == 0 else f"{n_fail} GATE(S) FAILED: report before using")
for _, r in gdf[gdf.status == "FAIL"].iterrows() if n_fail else []:
    print(f"   - [{r['scope']}] {r['gate']}: {r['detail']}")
print("=" * 78)

# ---- cell 33 ----
export = {
    "notebook": "W2",
    "generated_utc": RUN_STAMP,
    "environment": {"torch": torch.__version__, "timm": timm.__version__,
                    "device": DEVICE, "dtype": "float32",
                    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
                    "repo_path": str(repo)},
    "measurement": {"implementation": "repo metrics/ssdc.py evaluate_ssdc + main/model.py predict",
                    "RPI": True, "number_images": NUMBER_IMAGES, "batch_size": BATCH_SIZE,
                    "metric": METRIC, "num_workers": 0, "torch_seed": TORCH_SEED,
                    "mask_seed": MASK_SEED, "input_size": INPUT_SIZE,
                    "precision": "fp32 throughout; baseline and floor re-measured in-run"},
    "probe_settings": {"lr": PROBE_LR, "passes": PROBE_PASSES, "batch": PROBE_BATCH,
                       "train_frac": PROBE_TRAIN_FRAC, "split": "by image",
                       "standardised": True, "heads": ["exact", "row", "col"],
                       "reported": "peak test accuracy over passes"},
    "comparison_policy": "intra-model only",
    "results": RESULTS,
    "gates": GATES,
    "gate_summary": {"total": len(GATES),
                     "pass": int(sum(g["status"] == "PASS" for g in GATES)),
                     "fail": n_fail},
    "normalisation_protocol": {
        "formula": "(observed - floor) / (baseline - floor)",
        "source": "in-run fp32 baseline and floor from this notebook",
        "floors": {l: (e["floor_ssdc"][e["probe_layer"]]
                       if e.get("status") == "ok" and "floor_ssdc" in e else None)
                   for l, e in RESULTS.items()},
        "achievable_ranges": {l: e.get("achievable_range") for l, e in RESULTS.items()},
    },
}

out_json = OUT_DIR / f"w2_planes_{RUN_STAMP.replace(':', '-')}.json"
out_json.write_text(json.dumps(export, indent=2))
summary.to_csv(OUT_DIR / "w2_summary.csv", index=False)
gdf.to_csv(OUT_DIR / "w2_gate_report.csv", index=False)

per_block = []
for label, e in RESULTS.items():
    if e.get("status") != "ok":
        continue
    for cname, r in e.get("conditions", {}).items():
        if "ssdc" not in r:
            continue
        for bi, v in enumerate(r["ssdc"]):
            per_block.append({"run": label, "condition": cname, "block": bi, "ssdc": v})
pd.DataFrame(per_block).to_csv(OUT_DIR / "w2_per_block.csv", index=False)

import shutil
archive = shutil.make_archive("w2_output", "zip", OUT_DIR)
print("written:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p}  ({p.stat().st_size:,} bytes)")
print(f"\narchive: {archive} ({os.path.getsize(archive):,} bytes)")
try:
    from google.colab import files as cf
    cf.download(archive)
except Exception:
    print("(not Colab: copy the archive back manually)")
