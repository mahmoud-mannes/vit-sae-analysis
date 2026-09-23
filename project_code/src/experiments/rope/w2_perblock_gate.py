"""Gates for the per-block rope override used by W3.

1. Identity on every block through the per-block path must match the global override.
2. A prefix window must keep the blocks inside it at the floor and let later blocks
   recover; read after the window, since blocks inside it are at the floor by design.

Expected baseline and floor values are printed next to each measurement, so the
equivalence check cannot pass on two identical wrong measurements.
"""
import os, sys, json, time
from pathlib import Path

# ---------------------------------------------------------------- config
REPO_SRC   = os.environ.get("REPO_SRC", str(Path(__file__).resolve().parents[2]))
OUT_DIR    = os.environ.get("OUT_DIR", "perblock_out")
HF_HOME    = os.environ.get("HF_HOME")           # leave unset to use the default cache
N_IMAGES, BATCH, SEED = 1000, 256, 20260908
PROBE_L, WINDOW = 4, range(6)                    # mask blocks 0-5, read at PROBE_L and after
BASELINE_EXPECT, FLOOR_EXPECT = 0.4694, 0.0129   # naver at block 4, from W2
TOL = 0.01
# ------------------------------------------------------------------------

if HF_HOME:
    os.environ.setdefault("HF_HOME", HF_HOME)
import numpy as np, torch
sys.path.insert(0, REPO_SRC)
from metrics.ssdc import evaluate_ssdc
from main.load_models import load_rope, get_vit_blocks, get_block_attention
from experiments.common import load_imagenet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
DATASET = load_imagenet(split="validation", streaming=True)


def identity_like(emb):
    sin, cos = emb.chunk(2, -1)
    return torch.cat([torch.zeros_like(sin), torch.ones_like(cos)], -1)


class RopeGlobalOverride:
    """Wraps rope.get_embed, so it reaches every block. Same as in w2_rotary_plane_masking.py."""
    def __init__(self, model, transform):
        self.mod = next((m for _, m in model.named_modules()
                         if callable(getattr(m, "get_embed", None))), None)
        if self.mod is None:
            raise RuntimeError("no module exposing get_embed(): not a RoPE checkpoint?")
        self.transform = transform

    def __enter__(self):
        self.orig = self.mod.get_embed
        o, tf = self.orig, self.transform
        self.mod.get_embed = (lambda _o=o, _tf=tf: (lambda *a, **k: _tf(_o(*a, **k))))()
        return self

    def __exit__(self, *exc):
        self.mod.get_embed = self.orig


class RopePerBlockOverride:
    """Rewrites the rope kwarg on selected blocks only. Same as in w2_rotary_plane_masking.py."""
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


model, processor, source = load_rope(device=DEV, half=False, input_size=224)
model.eval().float()


def run(ctx=None):
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2 ** 31))
    kw = dict(RPI=True, number_images=N_IMAGES, batch_size=BATCH,
              metric="manhattan", half=False, num_workers=0)
    if ctx is None:
        scores, _ = evaluate_ssdc(model, processor, DATASET, source, **kw)
    else:
        with ctx:
            scores, _ = evaluate_ssdc(model, processor, DATASET, source, **kw)
    return [float(s) for s in scores]


GATES = []
def gate(name, passed, detail=""):
    GATES.append({"gate": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


t0 = time.time()
baseline   = run()
glob_floor = run(RopeGlobalOverride(model, identity_like))
per_floor  = run(RopePerBlockOverride(model, source, identity_like))
per_window = run(RopePerBlockOverride(model, source, identity_like, blocks=WINDOW))
last_masked, first_after = max(WINDOW), max(WINDOW) + 1

print()
gate("baseline is a real measurement, not an inert run",
     abs(baseline[PROBE_L] - BASELINE_EXPECT) < TOL,
     f"{baseline[PROBE_L]:+.6f} vs expected ~{BASELINE_EXPECT}")
gate("global override actually intervenes",
     abs(glob_floor[PROBE_L] - FLOOR_EXPECT) < TOL,
     f"{glob_floor[PROBE_L]:+.6f} vs expected ~{FLOOR_EXPECT}")

delta = max(abs(a - b) for a, b in zip(glob_floor, per_floor))
gate("per-block over all blocks equals the global override", delta < 1e-9,
     f"max |delta| over 12 blocks = {delta:.2e}")

# read after the window; inside it the value is the floor by design
inside_at_floor = abs(per_window[last_masked] - glob_floor[last_masked]) < 1e-6
recovers_after = per_window[-1] > glob_floor[-1] + 0.05
gate(f"masked window sits at the floor (block {last_masked})", inside_at_floor,
     f"{per_window[last_masked]:+.6f} vs floor {glob_floor[last_masked]:+.6f}")
gate(f"structure recovers after the window (block 11)", recovers_after,
     f"{per_window[-1]:+.6f} vs floor {glob_floor[-1]:+.6f}")

print(f"\nretention after masking blocks {min(WINDOW)}-{last_masked}, per block:")
print(f"  {'blk':>3s} {'baseline':>9s} {'floor':>8s} {'masked':>8s} {'retention':>10s}")
for i in range(len(baseline)):
    rng = baseline[i] - glob_floor[i]
    ret = (per_window[i] - glob_floor[i]) / rng if rng > 1e-6 else float("nan")
    tag = "  (in window)" if i in WINDOW else ""
    print(f"  {i:3d} {baseline[i]:9.4f} {glob_floor[i]:8.4f} {per_window[i]:8.4f} {ret:9.1%}{tag}")

out = os.path.join(OUT_DIR, "perblock_gate.json")
json.dump({"config": {"n_images": N_IMAGES, "batch": BATCH, "seed": SEED,
                      "window": list(WINDOW), "probe_layer": PROBE_L},
           "baseline": baseline, "global_floor": glob_floor,
           "perblock_all": per_floor, "perblock_window": per_window,
           "max_delta_global_vs_perblock": delta,
           "gates": GATES, "all_passed": all(g["status"] == "PASS" for g in GATES),
           "runtime_s": round(time.time() - t0, 1)}, open(out, "w"), indent=1)
print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")
