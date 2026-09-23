"""Probe protocol ablation on naver.

Varies feature standardisation and the train/test split (by image or by token) on the
same captured activations, to see which one accounts for the gap to LINEAR_PROBE.md.
"""
import os, sys, json, time, itertools
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.ssdc import evaluate_ssdc, _register_ssdc_accumulator_hooks
from main.load_models import load_rope, load_ape_timm, get_vit_blocks, get_block_attention
from experiments.common import load_imagenet

N, B, SEED = 1000, 256, 20260908
LR, PASSES, PBATCH = 5e-3, 20, 512          # LINEAR_PROBE.md hyperparameters
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DS = load_imagenet(split="validation", streaming=True)

class Cap:
    def __init__(s, n_prefix): s.n = n_prefix; s.buf = []
    def add(s, key, toks):
        if key == s.block: s.buf.append(toks.detach()[:, s.n:, :].float().cpu())

def capture(model, source, block, n_prefix):
    store = []
    attn = get_block_attention(get_vit_blocks(model, source)[block], source)
    h = attn.register_forward_hook(lambda m, i, o: store.append(i[0].detach()[:, n_prefix:, :].float().cpu()))
    torch.manual_seed(SEED); np.random.seed(SEED % (2**31))
    evaluate_ssdc(model, proc, DS, source, RPI=True, number_images=N, batch_size=B,
                  metric="manhattan", half=False, num_workers=0)
    h.remove()
    return torch.cat(store, 0)          # [n_img, T, D]

def probe(acts, n_classes, labels, standardise, split_by, seed=SEED):
    n_img, T, D = acts.shape
    feats = acts.reshape(-1, D)
    y = labels.repeat(n_img)
    g = torch.Generator().manual_seed(seed)
    if split_by == "image":
        idx = torch.randperm(n_img, generator=g); ntr = int(round(n_img * 0.8))
        rows = lambda ids: (ids.unsqueeze(1) * T + torch.arange(T).unsqueeze(0)).reshape(-1)
        tr, te = rows(idx[:ntr]), rows(idx[ntr:])
    else:                                # token-level split, ignores image boundaries
        perm = torch.randperm(n_img * T, generator=g); ntr = int(round(n_img * T * 0.8))
        tr, te = perm[:ntr], perm[ntr:]
    Xtr, ytr, Xte, yte = feats[tr].to(DEV), y[tr].to(DEV), feats[te].to(DEV), y[te].to(DEV)
    if standardise:
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    head = nn.Linear(D, n_classes).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    peak, n = 0.0, Xtr.shape[0]
    for _ in range(PASSES):
        p = torch.randperm(n, device=DEV)
        head.train()
        for s in range(0, n, PBATCH):
            b = p[s:s+PBATCH]; opt.zero_grad()
            F.cross_entropy(head(Xtr[b]), ytr[b]).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            acc = sum((head(Xte[s:s+4096]).argmax(-1) == yte[s:s+4096]).sum().item()
                      for s in range(0, Xte.shape[0], 4096)) / Xte.shape[0]
        peak = max(peak, acc)
    del Xtr, Xte, ytr, yte, head; torch.cuda.empty_cache()
    return peak

out = {}
for label, loader, block, ref in [
        ("RoPE · naver",  lambda: load_rope(device=DEV, half=False, input_size=224), 4, 38.70),
        ("APE · DINOv1",  lambda: load_ape_timm(device=DEV, half=False, input_size=224), 2, 95.90)]:
    model, proc, src = loader(); model.eval().float()
    npref = int(getattr(model, "num_prefix_tokens", 1))
    acts = capture(model, src, block, npref)
    T = acts.shape[1]; pos = torch.arange(T)
    print(f"\n=== {label}  block {block}  tokens {T}  (LINEAR_PROBE.md: {ref}%) ===", flush=True)
    res = {}
    for std, sp in itertools.product([True, False], ["image", "token"]):
        a = probe(acts, T, pos, std, sp) * 100
        tag = f"std={'Y' if std else 'N'} split={sp:5s}"
        res[tag] = a
        mark = "  <- 우리 프로토콜" if (std and sp == "image") else ""
        print(f"  {tag}  exact {a:6.2f}%   (vs {ref}%, delta {a-ref:+.2f}){mark}", flush=True)
    out[label] = {"reference": ref, "variants": res}
    del model, acts; torch.cuda.empty_cache()

json.dump(out, open("w2_output/probe_protocol_ablation.json", "w"), indent=1)
print("\nwrote w2_output/probe_protocol_ablation.json")
