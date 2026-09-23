import subprocess, sys

# ---------------- cell 2 ----------------
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

# ---------------- cell 3 ----------------
import os, json, math, time, warnings

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import timm

warnings.filterwarnings("ignore", category=UserWarning)

print("torch  :", torch.__version__)
print("timm   :", timm.__version__)
print("cuda   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device :", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"memory : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

# ---------------- cell 5 ----------------
# ============================================================================
# CONFIG
# ============================================================================
REPO_URL      = None  # clone fallback; not needed inside the repo
REPO_PATH     = str(Path(__file__).resolve().parents[2])
#               metrics/ and main/ live under project_code/src, not the repo root

NUMBER_IMAGES = 1000            # forwarded to evaluate_ssdc (matches its own default)
BATCH_SIZE    = 256             # forwarded to evaluate_ssdc (matches its own default)
METRIC        = "manhattan"     # repo default
TORCH_SEED    = 20260908        # reseeded before every pass; see note below

W1A_JSON      = None            # path to w1a_results_*.json; None = auto-search
CALIB_TOL     = 0.05            # tolerance for the calibration gate

OUT_DIR       = Path("w1b_output")

# load_ape()/load_rope() cast to fp16 whenever half=True, so tie it to CUDA.
HALF = torch.cuda.is_available()

# kind is passed to load_models.load_model, which picks the checkpoint and the source.
MODELS = {
    "APE · google ViT": dict(kind="ape",    family="APE",  role="primary",     probe_layer=2),
    "RoPE · naver":     dict(kind="rope",   family="RoPE", role="primary",     probe_layer=4),
    "APE · DINOv1":     dict(kind="dinov1", family="APE",  role="second_seed", probe_layer=2),
    "RoPE · DINOv3":    dict(kind="dinov3", family="RoPE", role="second_seed", probe_layer=4),
}
# ============================================================================

OUT_DIR.mkdir(exist_ok=True)
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GATES = []
def gate(name, scope, passed, detail=""):
    GATES.append({"gate": name, "scope": scope,
                  "status": "PASS" if passed else "FAIL", "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {scope}: {name}" + (f": {detail}" if detail else ""))
    return passed

print(f"device {DEVICE} | half {HALF} | images {NUMBER_IMAGES} | batch {BATCH_SIZE} | run {RUN_STAMP}")

# ---------------- cell 7 ----------------
def resolve_repo_path():
    '''Find a checkout with metrics/ and main/ as direct children, cloning if needed.'''
    candidates = []
    if REPO_PATH:
        candidates.append(Path(REPO_PATH).expanduser().resolve())
    # a bare clone puts project_code/src as the actual package root for this repo
    clone_dir = Path("vit-sae-analysis")
    candidates += [clone_dir / "project_code" / "src", clone_dir / "src", clone_dir]

    for c in candidates:
        if (c / "metrics").exists() and (c / "main").exists():
            return c

    if REPO_URL and not clone_dir.exists():
        print(f"cloning {REPO_URL} ...")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)])
        for c in candidates:
            if (c / "metrics").exists() and (c / "main").exists():
                return c

    raise FileNotFoundError(
        "Could not find a checkout with metrics/ and main/ as direct children. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". If the repo is private, clone it manually and set REPO_PATH."
    )

repo = resolve_repo_path()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
print(f"repo root: {repo}")

from metrics.ssdc import (
    evaluate_ssdc,
    spatial_similarity_distance_correlation,
    SSDCAccumulator,
)
from main.load_models import get_vit_blocks, get_block_attention, load_model
from main.model import predict
from main.prep_data import prep_data

print("imported evaluate_ssdc, load_model, predict, prep_data from the repo")

# The repo's documented unit-test property: similarity == negative distance -> +1
from scipy.spatial.distance import cdist as _cdist
_G = 14
_coords = np.stack(np.meshgrid(np.arange(_G), np.arange(_G), indexing="ij"), -1).reshape(-1, 2)
_D = _cdist(_coords, _coords, metric="cityblock")
_r_pos = spatial_similarity_distance_correlation(-_D, grid_size=_G, metric="manhattan")
_r_neg = spatial_similarity_distance_correlation(_D, grid_size=_G, metric="manhattan")
_r_rand = spatial_similarity_distance_correlation(
    np.random.default_rng(0).normal(size=(_G * _G, _G * _G)), grid_size=_G, metric="manhattan")
print(f"unit-test: S=-D -> {_r_pos:+.4f} | S=+D -> {_r_neg:+.4f} | random -> {_r_rand:+.4f}")
gate("imported metric satisfies its unit test", "repo-metric",
     _r_pos > 0.999 and _r_neg < -0.999 and abs(_r_rand) < 0.1,
     f"{_r_pos:+.3f} / {_r_neg:+.3f} / {_r_rand:+.3f}")

# ---------------- cell 9 ----------------
def load_val_dataset():
    """Validation split through experiments.common.load_imagenet: the gated ILSVRC split if
    the HF token can read it, otherwise the benjamin-paine/imagenet-1k-256x256 repack
    (same 0-999 label order)."""
    from experiments.common import load_imagenet
    return load_imagenet(split="validation", streaming=True)


DATASET = load_val_dataset()
print("dataset ready:", type(DATASET).__name__)

# ---------------- cell 11 ----------------
def mask_rope_planes(emb, kill):
    '''Set the listed rotation planes to identity (sin=0, cos=1). kill=[] must be a no-op.'''
    if not kill:
        return emb
    sin, cos = emb.chunk(2, -1)
    sin, cos = sin.clone(), cos.clone()
    for p in kill:
        sin[..., 2 * p:2 * p + 2] = 0.0
        cos[..., 2 * p:2 * p + 2] = 1.0
    return torch.cat([sin, cos], -1)


def identity_like(emb):
    sin, cos = emb.chunk(2, -1)
    return torch.cat([torch.zeros_like(sin), torch.ones_like(cos)], -1)


def find_rope_modules(model):
    '''Any submodule exposing get_embed(): covers .rope and rope_mixed layouts.'''
    found = []
    for name, mod in model.named_modules():
        if callable(getattr(mod, "get_embed", None)):
            found.append((name, mod))
    return found


class RopeOverride:
    def __init__(self, model, transform):
        self.mods = find_rope_modules(model)
        self.transform = transform
        self.orig = []

    def __enter__(self):
        if not self.mods:
            raise RuntimeError("no module with get_embed() found: not a RoPE checkpoint?")
        for name, mod in self.mods:
            o = mod.get_embed
            self.orig.append((mod, o))
            tf = self.transform
            mod.get_embed = (lambda _o=o, _tf=tf: (lambda *a, **k: _tf(_o(*a, **k))))()
        return self

    def __exit__(self, *exc):
        for mod, o in self.orig:
            mod.get_embed = o


def find_pos_embed(model):
    cands = [n for n, _ in model.named_parameters()
             if n.endswith("pos_embed") or n.endswith("position_embeddings")]
    if not cands:
        raise RuntimeError("no positional-embedding parameter found: not an APE checkpoint?")
    name = sorted(cands, key=len)[0]
    mod = model
    parts = name.split(".")
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return name, mod, parts[-1]


class APEZero:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.name, self.parent, self.attr = find_pos_embed(self.model)
        pe = getattr(self.parent, self.attr)
        self.saved = pe.detach().clone()
        with torch.no_grad():
            pe.zero_()
        print(f"    zeroed {self.name} {tuple(pe.shape)}")
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            getattr(self.parent, self.attr).copy_(self.saved)


# algebraic gates (no model required)
_e = torch.randn(196, 128)
gate("empty mask is bit-identical", "rope-algebra", torch.equal(mask_rope_planes(_e, []), _e))
gate("full mask equals identity", "rope-algebra",
     torch.allclose(mask_rope_planes(_e, list(range(32))), identity_like(_e)))

def _rot(x):
    x = x.unflatten(-1, (-1, 2))
    return torch.stack([-x[..., 1], x[..., 0]], -1).flatten(-2)
def _apply(x, emb):
    s, c = emb.chunk(2, -1)
    return x * c + _rot(x) * s
_x = torch.randn(2, 196, 64)
gate("identity embedding is a no-op on tokens", "rope-algebra",
     torch.allclose(_apply(_x, identity_like(_e)), _x, atol=1e-6))
gate("a real embedding does change tokens", "rope-algebra",
     not torch.allclose(_apply(_x, _e), _x))

# ---------------- cell 13 ----------------
def load_w1a():
    if W1A_JSON and Path(W1A_JSON).exists():
        return json.load(open(W1A_JSON))
    for pat in ["w1a_results_*.json", "w1a_output/w1a_results_*.json", "**/w1a_results_*.json"]:
        hits = sorted(Path(".").glob(pat))
        if hits:
            print("W1-A reference:", hits[-1])
            return json.load(open(hits[-1]))
    print("W1-A reference NOT FOUND: calibration gate will be skipped (not recommended)")
    return None

W1A = load_w1a()

def w1a_baseline(label, layer):
    if not W1A:
        return None
    for row in W1A.get("artifact_diagnostic", []):
        if row.get("run") == label and row.get("block") == layer:
            return row.get("untouched")
    return None

REFS = {lab: w1a_baseline(lab, c["probe_layer"]) for lab, c in MODELS.items()}
print("\nW1-A untouched baselines at probe layer:")
for k, v in REFS.items():
    print(f"  {k:20s} {'n/a' if v is None else f'{v:+.4f}'}")

# ---------------- cell 15 ----------------
def run_ssdc(model, processor, source):
    '''Baseline/floor SSDC-under-RPI, via the repo's own evaluate_ssdc. num_workers=0 for
    deterministic image order, which the empty-mask gate needs.'''
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED % (2**31))
    t0 = time.time()
    scores, _maps = evaluate_ssdc(
        model, processor, DATASET, source,
        RPI=True, number_images=NUMBER_IMAGES, batch_size=BATCH_SIZE,
        metric=METRIC, half=HALF, num_workers=0,
    )
    print(f"    ssdc: {len(scores)} blocks in {time.time()-t0:.0f}s")
    return [float(s) for s in scores]


def run_accuracy(model, processor, source):
    '''Clean-order and RPI-order top-1, via the repo's own predict(). Uses the same seed
    and num_workers=0 as run_ssdc so the RPI permutation sequence matches.'''
    accs = {}
    for tag, rpi in (("clean", False), ("rpi", True)):
        torch.manual_seed(TORCH_SEED)
        dl = prep_data(DATASET, processor, source, corruption_type=None,
                       number_images=NUMBER_IMAGES, batch_size=BATCH_SIZE,
                       half=HALF, num_workers=0)
        accs[tag] = float(predict(model, dl, source, RPI=rpi, half=HALF))
    return accs


RESULTS = {}

for label, cfg in MODELS.items():
    print("\n" + "=" * 78)
    print(f"{label}   (kind={cfg['kind']!r})")
    print("=" * 78)
    t_model = time.time()

    try:
        model, processor, source = load_model(cfg["kind"], device=DEVICE, half=HALF)
        model.eval()
    except Exception as e:
        gate("model loads", label, False, f"{type(e).__name__}: {e}")
        RESULTS[label] = {"status": "load_failed", "error": str(e)}
        continue

    n_blocks = len(get_vit_blocks(model, source))
    entry = {"status": "ok", "kind": cfg["kind"], "source": source,
             "family": cfg["family"], "role": cfg["role"],
             "probe_layer": cfg["probe_layer"], "n_blocks": n_blocks,
             "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", -1))}
    print(f"  source={source}  blocks={n_blocks}  prefix_tokens={entry['num_prefix_tokens']}")
    L = cfg["probe_layer"]

    # 1. baseline: SSDC + accuracy
    try:
        entry["baseline_ssdc"] = run_ssdc(model, processor, source)
        entry["baseline_acc"] = run_accuracy(model, processor, source)
        print(f"  baseline SSDC @ block {L}: {entry['baseline_ssdc'][L]:+.4f}"
              f"   top1 clean={entry['baseline_acc']['clean']:.3f} "
              f"rpi={entry['baseline_acc']['rpi']:.3f}")
    except Exception as e:
        gate("baseline measurement runs", label, False, f"{type(e).__name__}: {e}")
        RESULTS[label] = entry
        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        continue

    # 2. empty-mask gate (RoPE only)
    if cfg["family"] == "RoPE":
        try:
            mods = find_rope_modules(model)
            entry["rope_modules"] = [n for n, _ in mods]
            # RotaryEmbeddingDinoV3 keeps feat_shape=None even after a forward pass,
            # so the no-arg call asserts. Pass the grid explicitly in that case.
            try:
                emb0 = mods[0][1].get_embed()
            except (AssertionError, TypeError):
                emb0 = mods[0][1].get_embed(shape=tuple(model.patch_embed.grid_size))
            entry["n_planes"] = int(emb0.shape[-1] // 4)
            print(f"  rope modules {entry['rope_modules']}  planes={entry['n_planes']}")
            with RopeOverride(model, lambda e: mask_rope_planes(e, [])):
                empty = run_ssdc(model, processor, source)
            delta = float(np.max(np.abs(np.array(empty) - np.array(entry["baseline_ssdc"]))))
            gate("empty-mask reproduces baseline (E2.0)", label, delta < 1e-6,
                 f"max delta {delta:.2e}")
            entry["empty_mask_ssdc"] = empty
        except Exception as e:
            gate("empty-mask reproduces baseline (E2.0)", label, False, f"{type(e).__name__}: {e}")

    # 3. floor: SSDC + accuracy
    try:
        ctx = RopeOverride(model, identity_like) if cfg["family"] == "RoPE" else APEZero(model)
        with ctx:
            entry["floor_ssdc"] = run_ssdc(model, processor, source)
            entry["floor_acc"] = run_accuracy(model, processor, source)
        print(f"  floor    SSDC @ block {L}: {entry['floor_ssdc'][L]:+.4f}"
              f"   top1 clean={entry['floor_acc']['clean']:.3f} "
              f"rpi={entry['floor_acc']['rpi']:.3f}")
    except Exception as e:
        gate("floor measurement runs", label, False, f"{type(e).__name__}: {e}")
        RESULTS[label] = entry
        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        continue

    # 4. floor-drops gate
    b, f = entry["baseline_ssdc"][L], entry["floor_ssdc"][L]
    entry["achievable_range"] = b - f
    gate("floor is below baseline", label, f < b - 0.01,
         f"baseline {b:+.4f} -> floor {f:+.4f} (range {b-f:+.4f})")

    # 5. calibration gate
    ref = REFS.get(label)
    if ref is None:
        print("  [SKIP] calibration: no W1-A reference")
    else:
        d = abs(b - ref)
        gate("baseline matches W1-A", label, d <= CALIB_TOL,
             f"computed {b:+.4f} vs W1-A {ref:+.4f} (delta {d:.4f}, tol {CALIB_TOL})")
        entry["w1a_reference"] = ref
        entry["calibration_delta"] = float(d)

    entry["runtime_s"] = round(time.time() - t_model, 1)
    RESULTS[label] = entry
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"  done in {entry['runtime_s']}s")

# ---------------- cell 17 ----------------
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

rows = []
for label, e in RESULTS.items():
    if e.get("status") != "ok" or "floor_ssdc" not in e:
        rows.append({"run": label, "status": e.get("status", "incomplete")})
        continue
    L = e["probe_layer"]
    b, f = e["baseline_ssdc"][L], e["floor_ssdc"][L]
    rng = b - f
    bacc, facc = e.get("baseline_acc", {}), e.get("floor_acc", {})
    rows.append({"run": label, "status": "ok", "family": e["family"], "block": L,
                 "baseline": b, "floor": f, "achievable_range": rng,
                 "raw_drop_0.060_normalised": (0.060 / rng) if rng > 0 else np.nan,
                 "drop_needed_for_30pct": 0.30 * rng if rng > 0 else np.nan,
                 "baseline_top1_clean": bacc.get("clean"), "baseline_top1_rpi": bacc.get("rpi"),
                 "floor_top1_clean": facc.get("clean"), "floor_top1_rpi": facc.get("rpi"),
                 "w1a_ref": e.get("w1a_reference"),
                 "calib_delta": e.get("calibration_delta")})
summary = pd.DataFrame(rows)
print("FLOOR SUMMARY\n")
print(summary.to_string(index=False))
print()
for _, r in summary.iterrows():
    if r["status"] != "ok":
        continue
    if "DINOv1" in r["run"] or "DINOv3" in r["run"]:
        c = r.get("baseline_top1_clean")
        if c is not None and c < 0.02:
            print(f"  note: {r['run']} baseline clean top-1 = {c:.1%}: expected for a "
                  f"self-supervised checkpoint with no ImageNet-1k-trained head, not a bug.")

detail_rows = []
for label, e in RESULTS.items():
    if e.get("status") != "ok" or "floor_ssdc" not in e:
        continue
    for b in range(e["n_blocks"]):
        detail_rows.append({"run": label, "block": b,
                            "baseline": e["baseline_ssdc"][b],
                            "floor": e["floor_ssdc"][b],
                            "range": e["baseline_ssdc"][b] - e["floor_ssdc"][b]})
detail = pd.DataFrame(detail_rows)
print("\n\nPER-BLOCK DETAIL\n")
print(detail.to_string(index=False) if len(detail) else "(nothing to show)")

# ---------------- cell 18 ----------------
print("DOES FLOOR-NORMALISATION RESCUE DINOv3?\n")
d3 = summary[summary.run.str.contains("DINOv3")]
if len(d3) and d3.iloc[0].get("status") == "ok":
    r = d3.iloc[0]
    rng = r["achievable_range"]
    norm = 0.060 / rng if rng > 0 else float("nan")
    print(f"  baseline @ block 4     : {r['baseline']:+.4f}")
    print(f"  floor    @ block 4     : {r['floor']:+.4f}")
    print(f"  achievable range       : {rng:+.4f}")
    print(f"  raw max drop (22 feat) : 0.0599")
    print(f"  normalised damage      : {norm:.1%}\n")
    if norm >= 0.30:
        print("  -> CLEARS the 30% gate. Floor-normalisation rescues DINOv3.")
    else:
        print("  -> STILL FAILS the 30% gate. Floor-normalisation does not rescue DINOv3;")
        print("     this is the branch where Fisher z becomes the fallback to evaluate.")
else:
    print("  DINOv3 did not complete: cannot answer.")

# ---------------- cell 19 ----------------
ok = [(l, e) for l, e in RESULTS.items() if e.get("status") == "ok" and "floor_ssdc" in e]
if ok:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(ok), figsize=(4.2 * len(ok), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, e) in zip(axes, ok):
        x = range(e["n_blocks"])
        ax.plot(x, e["baseline_ssdc"], "o-", lw=1.9, ms=4, label="baseline")
        ax.plot(x, e["floor_ssdc"], "s--", lw=1.9, ms=4, color="crimson", label="floor")
        ax.fill_between(x, e["floor_ssdc"], e["baseline_ssdc"], alpha=0.15)
        ax.axvline(e["probe_layer"], color="k", ls=":", lw=1.2, alpha=0.7)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("block")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("SSDC under RPI")
    axes[0].legend(fontsize=8)
    fig.suptitle("Achievable range: shaded area is what normalisation divides by", y=1.02)
    fig.tight_layout()
    plt.show()
else:
    print("nothing to plot")

# ---------------- cell 21 ----------------
gdf = pd.DataFrame(GATES)
print("GATE REPORT\n")
print(gdf.to_string(index=False) if len(gdf) else "(none)")
n_fail = int((gdf.status == "FAIL").sum()) if len(gdf) else 0
print("\n" + "=" * 78)
if n_fail == 0:
    print("ALL GATES PASSED: floors are usable.")
else:
    print(f"{n_fail} GATE(S) FAILED: do not use these floors until resolved.")
    for _, r in gdf[gdf.status == "FAIL"].iterrows():
        print(f"   - [{r['scope']}] {r['gate']}: {r['detail']}")
print("=" * 78)

# ---------------- cell 23 ----------------
export = {
    "notebook": "W1-B",
    "generated_utc": RUN_STAMP,
    "environment": {"torch": torch.__version__, "timm": timm.__version__,
                    "device": DEVICE, "half": HALF,
                    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
                    "repo_path": str(repo)},
    "measurement": {"implementation": "repo metrics/ssdc.py evaluate_ssdc + main/model.py predict",
                    "RPI": True, "number_images": NUMBER_IMAGES,
                    "batch_size": BATCH_SIZE, "metric": METRIC,
                    "num_workers": 0, "torch_seed": TORCH_SEED,
                    "dataset_loader": "see console output for which loader was used: "
                                      "repo-specific if found under main/, else the "
                                      "unconfirmed imagenet-1k fallback"},
    "calibration_tolerance": CALIB_TOL,
    "results": RESULTS,
    "gates": GATES,
    "all_gates_passed": bool(n_fail == 0),
    "normalisation_protocol": {
        "formula": "(observed - floor) / (baseline - floor)",
        "floors": {l: (e["floor_ssdc"][e["probe_layer"]]
                       if e.get("status") == "ok" and "floor_ssdc" in e else None)
                   for l, e in RESULTS.items()},
        "achievable_ranges": {l: e.get("achievable_range") for l, e in RESULTS.items()},
    },
}

out_json = OUT_DIR / f"w1b_floors_{RUN_STAMP.replace(':', '-')}.json"
out_json.write_text(json.dumps(export, indent=2))
summary.to_csv(OUT_DIR / "floor_summary.csv", index=False)
if len(detail):
    detail.to_csv(OUT_DIR / "floor_per_block.csv", index=False)
gdf.to_csv(OUT_DIR / "gate_report.csv", index=False)

import shutil
archive = shutil.make_archive("w1b_output", "zip", OUT_DIR)
print("written:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p}  ({p.stat().st_size:,} bytes)")
print(f"\narchive: {archive} ({os.path.getsize(archive):,} bytes)")

try:
    from google.colab import files as cf
    cf.download(archive)
except Exception:
    print("(not Colab: copy the archive back manually)")
