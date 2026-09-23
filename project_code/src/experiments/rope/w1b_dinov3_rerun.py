# DINOv3 only: repo's load_model hardcodes a 'timm/'-prefixed name that current timm rejects.
# Call load_rope directly with the registry name, then merge into the existing W1-B output.
import os, sys, json, time, glob
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics.ssdc import evaluate_ssdc
from main.load_models import get_vit_blocks, load_rope
from main.model import predict
from main.prep_data import prep_data
from experiments.common import load_imagenet

NUMBER_IMAGES, BATCH_SIZE, METRIC, TORCH_SEED = 1000, 256, "manhattan", 20260908
HALF = torch.cuda.is_available(); DEVICE = "cuda" if HALF else "cpu"
LABEL, L = "RoPE · DINOv3", 4
DATASET = load_imagenet(split="validation", streaming=True)

def identity_like(emb):
    sin, cos = emb.chunk(2, -1)
    return torch.cat([torch.zeros_like(sin), torch.ones_like(cos)], -1)

def mask_none(emb):
    return emb

class RopeOverride:
    def __init__(self, model, transform):
        self.mods = [(n, m) for n, m in model.named_modules() if callable(getattr(m, "get_embed", None))]
        self.transform = transform; self.orig = []
    def __enter__(self):
        for name, mod in self.mods:
            o = mod.get_embed; self.orig.append((mod, o)); tf = self.transform
            mod.get_embed = (lambda _o=o, _tf=tf: (lambda *a, **k: _tf(_o(*a, **k))))()
        return self
    def __exit__(self, *e):
        for mod, o in self.orig: mod.get_embed = o

def run_ssdc(model, processor, source):
    torch.manual_seed(TORCH_SEED); np.random.seed(TORCH_SEED % (2**31)); t0 = time.time()
    scores, _ = evaluate_ssdc(model, processor, DATASET, source, RPI=True,
                              number_images=NUMBER_IMAGES, batch_size=BATCH_SIZE,
                              metric=METRIC, half=HALF, num_workers=0)
    print(f"    ssdc: {len(scores)} blocks in {time.time()-t0:.0f}s", flush=True)
    return [float(s) for s in scores]

def run_accuracy(model, processor, source):
    out = {}
    for tag, rpi in (("clean", False), ("rpi", True)):
        torch.manual_seed(TORCH_SEED)
        dl = prep_data(DATASET, processor, source, corruption_type=None,
                       number_images=NUMBER_IMAGES, batch_size=BATCH_SIZE, half=HALF, num_workers=0)
        out[tag] = float(predict(model, dl, source, RPI=rpi, half=HALF))
    return out

GATES = []
def gate(name, passed, detail=""):
    GATES.append({"gate": name, "scope": LABEL, "status": "PASS" if passed else "FAIL", "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {LABEL}: {name}" + (f": {detail}" if detail else ""), flush=True)

t0 = time.time()
model, processor, source = load_rope(model_name="vit_base_patch16_dinov3.lvd1689m", device=DEVICE, half=HALF)
model.eval()
gate("model loads (registry name, no timm/ prefix)", True, "vit_base_patch16_dinov3.lvd1689m")

e = {"status": "ok", "kind": "dinov3", "source": source, "family": "RoPE", "role": "second_seed",
     "probe_layer": L, "n_blocks": len(get_vit_blocks(model, source)),
     "num_prefix_tokens": int(getattr(model, "num_prefix_tokens", -1)),
     "grid_size": list(model.patch_embed.grid_size),
     "note": "loaded with the timm registry name; the repo's load_model() hardcodes 'timm/...' which current timm rejects"}
print(f"  source={source} blocks={e['n_blocks']} prefix={e['num_prefix_tokens']} grid={e['grid_size']}", flush=True)

e["baseline_ssdc"] = run_ssdc(model, processor, source)
e["baseline_acc"]  = run_accuracy(model, processor, source)
print(f"  baseline SSDC @ block {L}: {e['baseline_ssdc'][L]:+.4f}   top1 clean={e['baseline_acc']['clean']:.3f} rpi={e['baseline_acc']['rpi']:.3f}", flush=True)

mods = [(n, m) for n, m in model.named_modules() if callable(getattr(m, "get_embed", None))]
emb0 = mods[0][1].get_embed(shape=tuple(model.patch_embed.grid_size))
e["rope_modules"] = [n for n, _ in mods]; e["n_planes"] = int(emb0.shape[-1] // 4)
print(f"  rope modules {e['rope_modules']} planes={e['n_planes']} emb={tuple(emb0.shape)}", flush=True)

with RopeOverride(model, mask_none):
    empty = run_ssdc(model, processor, source)
d = float(np.max(np.abs(np.array(empty) - np.array(e["baseline_ssdc"]))))
gate("empty-mask reproduces baseline (E2.0)", d < 1e-6, f"max delta {d:.2e}")
e["empty_mask_ssdc"] = empty

with RopeOverride(model, identity_like):
    e["floor_ssdc"] = run_ssdc(model, processor, source)
    e["floor_acc"]  = run_accuracy(model, processor, source)
print(f"  floor    SSDC @ block {L}: {e['floor_ssdc'][L]:+.4f}   top1 clean={e['floor_acc']['clean']:.3f} rpi={e['floor_acc']['rpi']:.3f}", flush=True)

b, f = e["baseline_ssdc"][L], e["floor_ssdc"][L]
e["achievable_range"] = b - f
gate("floor is below baseline", f < b - 0.01, f"baseline {b:+.4f} -> floor {f:+.4f} (range {b-f:+.4f})")

w1a = json.load(open(sorted(glob.glob("w1a_output/w1a_results_*.json"))[-1]))
ref = next((r["untouched"] for r in w1a["artifact_diagnostic"] if r["run"] == LABEL and r["block"] == L), None)
if ref is not None:
    gate("baseline matches W1-A", abs(b - ref) <= 0.05, f"computed {b:+.4f} vs W1-A {ref:+.4f} (delta {abs(b-ref):.4f}, tol 0.05)")
    e["w1a_reference"] = ref; e["calibration_delta"] = float(abs(b - ref))
e["runtime_s"] = round(time.time() - t0, 1)

out = sorted(glob.glob("w1b_output/w1b_floors_*.json"))[-1]
doc = json.load(open(out))
doc["results"][LABEL] = e
doc.setdefault("gates", []).extend(GATES)
doc["dinov3_rerun_note"] = e["note"]
json.dump(doc, open(out, "w"), indent=1)
print(f"\nmerged into {out}  ({e['runtime_s']}s)", flush=True)
