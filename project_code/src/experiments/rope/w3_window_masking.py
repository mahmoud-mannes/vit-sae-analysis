import subprocess, sys

# ---- cell 2 ----
import importlib, subprocess, sys
def ensure(pkg, mod=None):
    try:
        importlib.import_module(mod or pkg); return False
    except ImportError:
        print(f"installing {pkg} ..."); subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg]); return True
for p in ["timm","transformers","datasets","scipy"]:
    ensure(p)
print("environment ready")

# ---- cell 3 ----
import os
import os, json, math, time, warnings
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
    if free < 15e9:
        print("  note: low headroom. If probe training OOMs, lower PROBE_CHUNK in the config.")

# ---- cell 5 ----
# ============================================================================
# CONFIG
# ============================================================================
REPO_URL   = None
REPO_PATH  = str(Path(__file__).resolve().parents[2])

NUMBER_IMAGES = 1000
BATCH_SIZE    = 256
METRIC        = "manhattan"
TORCH_SEED    = 20260908
MASK_SEED     = 771
INPUT_SIZE    = 224

PROBE_LR, PROBE_PASSES, PROBE_BATCH, PROBE_TRAIN_FRAC = 5e-3, 20, 512, 0.8
PROBE_CHUNK   = 8192     # eval chunk size
PROBE_DEVICE  = None     # None = same as model device; "cpu" to force probes onto CPU

ANCHOR_TOL    = 0.01     # |measured - W2 anchor| must be under this
FLOOR_TOL     = 1e-3     # in-window floor agreement, prefix conditions only
EQUIV_TOL     = 1e-9     # per-block vs global identity
PROBE_CHANCE_TOL = 0.03  # "at chance" band for a masked axis probe
PROBE_BASE_TOL   = 0.10  # "at baseline" band for a surviving axis probe

PREFIX_KS = [0,1,2,3,4,5,6,7,8,9]
SUFFIX_KS = [4,6,8]          # readout starts at k+1, so low k gives more points
AXISROW_KS = [2,5,8]         # k=5 is the comparison anchor
AXISCOL_KS = [5]
RAND_KS    = [5]
MIDDLE     = (4,7)

# E1.2 combined SDs at the probe layer, for the recovery-onset threshold.
NOISE_SD = {"RoPE · naver": 0.00834, "RoPE · DINOv3": 0.00077}
ONSET_K  = 3.0

# W2 fp32 measurements, used as absolute gate anchors.
W2_ANCHORS = {
    "APE · google ViT": dict(probe_layer=2, baseline=0.7406, floor=0.0151,
                             probe_exact=1.0000, probe_row=1.0000, probe_col=1.0000),
    "RoPE · naver":     dict(probe_layer=4, baseline=0.4694, floor=0.0129,
                             probe_exact=0.3265, probe_row=0.6536, probe_col=0.6268),
    "APE · DINOv1":     dict(probe_layer=2, baseline=0.4700, floor=0.0114,
                             probe_exact=0.9998, probe_row=0.9999, probe_col=0.9999),
    "RoPE · DINOv3":    dict(probe_layer=4, baseline=0.9487, floor=0.0177,
                             probe_exact=0.3959, probe_row=0.7561, probe_col=0.7665),
}

MODELS = {
    "APE · google ViT": dict(loader="load_ape",      kwargs={},                                  family="APE"),
    "RoPE · naver":     dict(loader="load_rope",     kwargs=dict(input_size=INPUT_SIZE),         family="RoPE"),
    "APE · DINOv1":     dict(loader="load_ape_timm", kwargs=dict(input_size=INPUT_SIZE),         family="APE"),
    "RoPE · DINOv3":    dict(loader="load_rope",
                             kwargs=dict(model_name="vit_base_patch16_dinov3.lvd1689m",
                                         input_size=INPUT_SIZE),                                 family="RoPE"),
}

OUT_DIR = Path("w3_output"); OUT_DIR.mkdir(exist_ok=True)
# ============================================================================

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDEV = PROBE_DEVICE or DEVICE
N_BLOCKS = 12

GATES = []
def gate(name, scope, passed, detail=""):
    GATES.append({"gate":name,"scope":scope,"status":"PASS" if passed else "FAIL","detail":detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {scope}: {name}" + (f": {detail}" if detail else ""))
    return passed

print(f"device {DEVICE} | probes on {PDEV} | fp32 | images {NUMBER_IMAGES} | batch {BATCH_SIZE}")
print(f"run {RUN_STAMP}")

# ---- cell 7 ----
def resolve_repo_path():
    cands = []
    if REPO_PATH: cands.append(Path(REPO_PATH).expanduser().resolve())
    clone = Path("vit-sae-analysis")
    cands += [clone/"project_code"/"src", clone/"src", clone]
    for c in cands:
        if (c/"metrics").exists() and (c/"main").exists(): return c
    if REPO_URL and not clone.exists():
        print(f"cloning {REPO_URL} ...")
        subprocess.check_call(["git","clone","--depth","1",REPO_URL,str(clone)])
        for c in cands:
            if (c/"metrics").exists() and (c/"main").exists(): return c
    raise FileNotFoundError("No checkout with metrics/ and main/. Tried: "
                            + ", ".join(str(c) for c in cands))

repo = resolve_repo_path()
if str(repo) not in sys.path: sys.path.insert(0, str(repo))
print("repo root:", repo)

from metrics.ssdc import evaluate_ssdc, spatial_similarity_distance_correlation
from main.load_models import (get_vit_blocks, get_block_attention, get_patch_embed_conv,
                              load_ape, load_rope, load_ape_timm)
from main.model import predict
from main.prep_data import prep_data
from experiments.common import load_imagenet

LOADERS = {"load_ape":load_ape, "load_rope":load_rope, "load_ape_timm":load_ape_timm}
print("imported evaluate_ssdc, loaders, predict, prep_data, load_imagenet")

from scipy.spatial.distance import cdist as _cdist
_G=14
_c=np.stack(np.meshgrid(np.arange(_G),np.arange(_G),indexing="ij"),-1).reshape(-1,2)
_D=_cdist(_c,_c,metric="cityblock")
_a=spatial_similarity_distance_correlation(-_D,grid_size=_G,metric="manhattan")
_b=spatial_similarity_distance_correlation( _D,grid_size=_G,metric="manhattan")
_r=spatial_similarity_distance_correlation(np.random.default_rng(0).normal(size=(_G*_G,_G*_G)),
                                           grid_size=_G,metric="manhattan")
gate("imported metric satisfies its unit test","repo-metric",
     _a>0.999 and _b<-0.999 and abs(_r)<0.1, f"{_a:+.3f} / {_b:+.3f} / {_r:+.3f}")

DATASET = load_imagenet(split="validation", streaming=True)
print("dataset ready:", type(DATASET).__name__)

# ---- cell 9 ----
def find_rope_module(model):
    for name, mod in model.named_modules():
        if callable(getattr(mod, "get_embed", None)):
            return name, mod
    return None, None

def capture_forward_grid(model, processor, source):
    grid = {}
    conv = get_patch_embed_conv(model, source)
    h = conv.register_forward_hook(
        lambda m,i,o: grid.update(hw=(int(o.shape[2]), int(o.shape[3])) if o.dim()==4 else None))
    dl = prep_data(DATASET, processor, source, corruption_type=None,
                   number_images=BATCH_SIZE, batch_size=BATCH_SIZE, half=False, num_workers=0)
    with torch.no_grad():
        for images,_ in dl:
            if source=="transformers":
                model(**{k:v.to(DEVICE) for k,v in images.items()})
            else:
                model(images.to(DEVICE))
            break
    h.remove()
    return grid.get("hw")

def read_rotate_half(model, source):
    for blk in get_vit_blocks(model, source):
        attn = get_block_attention(blk, source)
        for attr in ("rotate_half","half","rope_rotate_half"):
            v = getattr(attn, attr, None)
            if isinstance(v, bool): return v, attr
    return False, "default(interleaved)"

def plane_columns(p, n_planes, rotate_half):
    return [p, p+n_planes] if rotate_half else [2*p, 2*p+1]

def derive_axis_map(emb, n_planes, rotate_half, H, W):
    '''Classify each plane as row- or column-carrying, empirically and wrap-safely.'''
    sin, cos = emb.detach().float().chunk(2, -1)
    rows, decisive = [], True
    for p in range(n_planes):
        c = plane_columns(p, n_planes, rotate_half)[0]
        z = torch.complex(cos[:,c], sin[:,c]).reshape(H,W)
        ra = (z - z.mean(dim=1,keepdim=True)).abs().mean().item()   # variation within a row
        rb = (z - z.mean(dim=0,keepdim=True)).abs().mean().item()   # variation within a col
        axis = "row" if ra < rb else "col"
        lo,hi = sorted([ra,rb]); ratio = lo/hi if hi>1e-12 else 1.0
        if ratio > 0.10: decisive = False
        rows.append({"plane":p,"axis":axis,"ratio":ratio})
    rp=[r["plane"] for r in rows if r["axis"]=="row"]
    cp=[r["plane"] for r in rows if r["axis"]=="col"]
    return rows, rp, cp, decisive

def freq_strength(emb, n_planes, rotate_half, H, W):
    '''Phase turn rate across the grid. Higher = higher spatial frequency.'''
    sin, cos = emb.detach().float().chunk(2,-1)
    out={}
    for p in range(n_planes):
        c = plane_columns(p, n_planes, rotate_half)[0]
        z = torch.complex(cos[:,c], sin[:,c]).reshape(H,W)
        dr=(z[1:,:]-z[:-1,:]).abs().mean().item()
        dc=(z[:,1:]-z[:,:-1]).abs().mean().item()
        out[p]=max(dr,dc)
    return out

# ---- cell 11 ----
def mask_planes(emb, kill, n_planes, rotate_half):
    if not kill: return emb
    sin, cos = emb.chunk(2,-1)
    sin, cos = sin.clone(), cos.clone()
    for p in kill:
        for c in plane_columns(p, n_planes, rotate_half):
            sin[...,c]=0.0; cos[...,c]=1.0
    return torch.cat([sin,cos],-1)

def identity_like(emb):
    sin,cos = emb.chunk(2,-1)
    return torch.cat([torch.zeros_like(sin), torch.ones_like(cos)],-1)

class RopeGlobalOverride:
    def __init__(self, model, transform):
        self.name, self.mod = find_rope_module(model)
        self.transform=transform; self.orig=None
    def __enter__(self):
        if self.mod is None: raise RuntimeError("no get_embed module: not a RoPE checkpoint?")
        self.orig=self.mod.get_embed
        o,tf=self.orig,self.transform
        self.mod.get_embed=(lambda _o=o,_tf=tf: (lambda *a,**k: _tf(_o(*a,**k))))()
        return self
    def __exit__(self,*e): self.mod.get_embed=self.orig

class RopePerBlockOverride:
    def __init__(self, model, source, transform, blocks=None):
        self.blocks=get_vit_blocks(model, source)
        self.which=list(range(len(self.blocks))) if blocks is None else list(blocks)
        self.source=source; self.transform=transform; self.handles=[]
    def __enter__(self):
        tf=self.transform
        def make(_i):
            def pre(module,args,kwargs):
                r=kwargs.get("rope",None)
                if r is not None:
                    kwargs=dict(kwargs); kwargs["rope"]=tf(r)
                return args,kwargs
            return pre
        for i in self.which:
            attn=get_block_attention(self.blocks[i], self.source)
            self.handles.append(attn.register_forward_pre_hook(make(i), with_kwargs=True))
        return self
    def __exit__(self,*e):
        for h in self.handles: h.remove()
        self.handles=[]

class APEZero:
    def __init__(self, model): self.model=model
    def __enter__(self):
        cands=[n for n,_ in self.model.named_parameters()
               if n.endswith("pos_embed") or n.endswith("position_embeddings")]
        if not cands: raise RuntimeError("no positional-embedding parameter found")
        self.name=sorted(cands,key=len)[0]; parts=self.name.split(".")
        mod=self.model
        for p in parts[:-1]: mod=getattr(mod,p)
        self.parent,self.attr=mod,parts[-1]
        pe=getattr(self.parent,self.attr); self.saved=pe.detach().clone()
        with torch.no_grad(): pe.zero_()
        return self
    def __exit__(self,*e):
        with torch.no_grad(): getattr(self.parent,self.attr).copy_(self.saved)

class NullCtx:
    def __enter__(self): return self
    def __exit__(self,*e): return False

# ---- cell 13 ----
def stratified_axis_matched(row_planes, col_planes, freq, n_per_axis, seed):
    '''Remove the same frequency bands from both axes, so the halves are matched exactly.'''
    rs = sorted(row_planes, key=lambda p: -freq[p])   # index 0 = highest frequency
    cs = sorted(col_planes, key=lambda p: -freq[p])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rs), size=n_per_axis, replace=False).tolist())
    return sorted([rs[i] for i in idx] + [cs[i] for i in idx]), idx

def _rot_il(x):
    x=x.unflatten(-1,(-1,2))
    return torch.stack([-x[...,1],x[...,0]],-1).flatten(-2)
def _apply_il(x,emb):
    s,c=emb.chunk(2,-1); return x*c+_rot_il(x)*s

for lname, rh in [("interleaved (naver)",False), ("rotate_half (DINOv3)",True)]:
    npl,dim = 32,64
    e=torch.randn(196,2*dim)
    gate(f"empty mask is bit-identical [{lname}]","rope-algebra", torch.equal(mask_planes(e,[],npl,rh),e))
    gate(f"full mask equals identity [{lname}]","rope-algebra",
         torch.allclose(mask_planes(e,list(range(npl)),npl,rh), identity_like(e)))
    m=mask_planes(e,[3],npl,rh)
    touched=torch.nonzero((m-e).abs().sum(0)>0).flatten().tolist()
    sc=plane_columns(3,npl,rh); expect=sorted(set(sc+[c+dim for c in sc]))
    gate(f"masking plane 3 touches only its own columns [{lname}]","rope-algebra",
         touched==expect, f"{touched} vs {expect}")

x=torch.randn(2,196,64); e=torch.randn(196,128)
gate("identity embedding is a no-op on tokens","rope-algebra",
     torch.allclose(_apply_il(x,identity_like(e)),x,atol=1e-6))
gate("a real embedding does change tokens","rope-algebra", not torch.allclose(_apply_il(x,e),x))

_rp,_cp=list(range(16)),list(range(16,32))
_f={p:(1.0/(1+p%16)) for p in range(32)}
_k,_idx=stratified_axis_matched(_rp,_cp,_f,8,MASK_SEED)
_rs=sorted(_rp,key=lambda p:-_f[p]); _cs=sorted(_cp,key=lambda p:-_f[p])
gate("frequency-matched control is axis-balanced","control",
     sum(1 for p in _k if p in _rp)==8 and sum(1 for p in _k if p in _cp)==8,
     f"{sum(1 for p in _k if p in _rp)}/{sum(1 for p in _k if p in _cp)} from bands {_idx}")
gate("frequency-matched control removes identical bands from both axes","control",
     sorted(i for i,p in enumerate(_rs) if p in set(_k))==sorted(i for i,p in enumerate(_cs) if p in set(_k))==sorted(_idx))

# ---- cell 15 ----
class Capture:
    def __init__(self, model, source, blocks, n_prefix):
        self.blocks=get_vit_blocks(model,source); self.which=list(blocks)
        self.n_prefix=int(n_prefix); self.source=source
        self.store={i:[] for i in self.which}; self.handles=[]
    def __enter__(self):
        def make(i):
            def hook(module, inputs, output):
                t=inputs[0].detach()
                if self.n_prefix>0: t=t[:, self.n_prefix:, :]
                self.store[i].append(t.float().cpu())
            return hook
        for i in self.which:
            attn=get_block_attention(self.blocks[i], self.source)
            self.handles.append(attn.register_forward_hook(make(i)))
        return self
    def __exit__(self,*e):
        for h in self.handles: h.remove()
        self.handles=[]
    def stacked(self,i):
        return torch.cat(self.store[i],0) if self.store[i] else None

def _fit_head(Xtr,ytr,Xte,yte,n_classes):
    head=nn.Linear(Xtr.shape[1], n_classes).to(Xtr.device)
    opt=torch.optim.AdamW(head.parameters(), lr=PROBE_LR)
    n=Xtr.shape[0]; peak=0.0
    for _ in range(PROBE_PASSES):
        perm=torch.randperm(n, device=Xtr.device); head.train()
        for s in range(0,n,PROBE_BATCH):
            b=perm[s:s+PROBE_BATCH]
            opt.zero_grad(); F.cross_entropy(head(Xtr[b]), ytr[b]).backward(); opt.step()
        head.eval(); correct=0
        with torch.no_grad():
            for s in range(0,Xte.shape[0],PROBE_CHUNK):
                correct+=(head(Xte[s:s+PROBE_CHUNK]).argmax(-1)==yte[s:s+PROBE_CHUNK]).sum().item()
        peak=max(peak, correct/Xte.shape[0])
    del head, opt
    return float(peak)

def probe_block(feats, H, W, n_images, seed=TORCH_SEED):
    '''All three heads from one transfer and one standardisation.'''
    T=H*W
    g=torch.Generator().manual_seed(seed)
    idx=torch.randperm(n_images, generator=g)
    n_tr=int(round(n_images*PROBE_TRAIN_FRAC))
    def rows(ids):
        return (ids.unsqueeze(1)*T + torch.arange(T).unsqueeze(0)).reshape(-1)
    tr,te=rows(idx[:n_tr]), rows(idx[n_tr:])

    Xtr=feats[tr].to(PDEV); Xte=feats[te].to(PDEV)
    mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True).clamp_min(1e-6)
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd

    slot=torch.arange(T)
    out={}
    for name, lab, nc in (("exact",slot,T), ("row",slot//W,H), ("col",slot%W,W)):
        y=lab.repeat(n_images)
        out[name]=_fit_head(Xtr, y[tr].to(PDEV), Xte, y[te].to(PDEV), nc)
    out["chance"]={"exact":1.0/T,"row":1.0/H,"col":1.0/W}
    del Xtr,Xte
    if PDEV=="cuda": torch.cuda.empty_cache()
    return out

# ---- cell 17 ----
def spec_for(name, kind, window, planes, n_planes):
    last = max(window) if window else None
    if kind=="prefix":     read_from, probe_from = last+2, last+2
    elif kind=="suffix":   read_from, probe_from = min(window)+1, min(window)+1
    elif kind=="middle":   read_from, probe_from = last+2, last+2
    elif kind in ("axis","rand"): read_from, probe_from = last+2, 1
    else:                  read_from, probe_from = 0, 0
    return dict(name=name, kind=kind, window=sorted(window) if window else [],
                planes=planes, read_from=read_from,
                probe_blocks=[b for b in range(probe_from, N_BLOCKS)])

def build_conditions(family, n_planes, row_planes, col_planes, freq):
    C=[spec_for("baseline","baseline",None,None,n_planes),
       spec_for("floor","floor",list(range(N_BLOCKS)),"ALL",n_planes)]
    if family!="RoPE":
        for c in C: c["probe_blocks"]=list(range(N_BLOCKS))
        return C, None
    C[0]["probe_blocks"]=list(range(N_BLOCKS))
    C[1]["probe_blocks"]=list(range(N_BLOCKS))
    for k in PREFIX_KS:
        C.append(spec_for(f"prefix_{k}","prefix",list(range(0,k+1)),"ALL",n_planes))
    for k in SUFFIX_KS:
        C.append(spec_for(f"suffix_{k}","suffix",list(range(k,N_BLOCKS)),"ALL",n_planes))
    C.append(spec_for(f"middle_{MIDDLE[0]}_{MIDDLE[1]}","middle",
                      list(range(MIDDLE[0],MIDDLE[1]+1)),"ALL",n_planes))
    for k in AXISROW_KS:
        C.append(spec_for(f"axisrow_{k}","axis",list(range(0,k+1)),list(row_planes),n_planes))
    for k in AXISCOL_KS:
        C.append(spec_for(f"axiscol_{k}","axis",list(range(0,k+1)),list(col_planes),n_planes))
    strat=None
    for k in RAND_KS:
        kill, bands = stratified_axis_matched(row_planes,col_planes,freq,n_planes//4,MASK_SEED)
        strat={"planes":kill,"bands":bands}
        C.append(spec_for(f"rand16_{k}","rand",list(range(0,k+1)),kill,n_planes))
    return C, strat

def make_ctx(model, source, spec, family, n_planes, rotate_half):
    if spec["kind"]=="baseline": return NullCtx()
    if family!="RoPE":           return APEZero(model)
    if spec["kind"]=="floor":    return RopeGlobalOverride(model, identity_like)
    if spec["planes"]=="ALL":    tf = identity_like
    else:
        kill=spec["planes"]
        tf = lambda e, k=kill: mask_planes(e, k, n_planes, rotate_half)
    return RopePerBlockOverride(model, source, tf, blocks=spec["window"])

def measure(model, processor, source, ctx, probe_blocks, n_prefix, H, W, tag):
    torch.manual_seed(TORCH_SEED); np.random.seed(TORCH_SEED % (2**31))
    t0=time.time()
    cap = Capture(model, source, probe_blocks, n_prefix) if probe_blocks else None
    if cap: cap.__enter__()
    try:
        with ctx:
            scores,_ = evaluate_ssdc(model, processor, DATASET, source,
                                     RPI=True, number_images=NUMBER_IMAGES,
                                     batch_size=BATCH_SIZE, metric=METRIC,
                                     half=False, num_workers=0)
    finally:
        if cap: cap.__exit__()
    res={"ssdc":[float(s) for s in scores], "probes":{}}
    if cap:
        for i in probe_blocks:
            acts=cap.stacked(i)
            if acts is None: continue
            res["probes"][str(i)] = probe_block(acts.reshape(-1,acts.shape[-1]), H, W, acts.shape[0])
            del acts
        cap.store.clear()
    res["hooks_registered"] = len(getattr(ctx, "handles", []) or [])
    res["runtime_s"]=round(time.time()-t0,1)
    return res

# ---- cell 19 ----
RESULTS={}

for label,cfg in MODELS.items():
    print("\n"+"="*78); print(f"{label}"); print("="*78)
    t_model=time.time()
    A=W2_ANCHORS[label]; PL=A["probe_layer"]

    try:
        model, processor, source = LOADERS[cfg["loader"]](device=DEVICE, half=False, **cfg["kwargs"])
        model.eval().float()
    except Exception as ex:
        gate("model loads", label, False, f"{type(ex).__name__}: {ex}")
        RESULTS[label]={"status":"load_failed","error":str(ex)}; continue

    n_prefix = 1 if source=="transformers" else int(getattr(model,"num_prefix_tokens",1))
    hw = capture_forward_grid(model, processor, source)
    if hw is None: raise RuntimeError("could not read the forward-pass grid")
    H,W = hw; T=H*W
    print(f"  source={source} prefix={n_prefix} grid={H}x{W} tokens={T}")
    gate("forward grid is 14x14", label, (H,W)==(14,14), f"{H}x{W}")

    entry={"status":"ok","family":cfg["family"],"source":source,"probe_layer":PL,
           "num_prefix_tokens":n_prefix,"grid":[H,W],"patch_tokens":T,"dtype":"float32",
           "n_blocks":len(get_vit_blocks(model,source))}
    if hasattr(model,"patch_embed"):
        gs=list(getattr(model.patch_embed,"grid_size",()) or [])
        entry["patch_embed_grid_size"]=gs
        if gs and tuple(gs)!=(H,W):
            print(f"    note: patch_embed.grid_size={gs} but forward grid={(H,W)}: using the forward pass")

    n_planes=rotate_half=None; row_planes=col_planes=[]; freq={}
    if cfg["family"]=="RoPE":
        rname,rmod = find_rope_module(model)
        rotate_half, rh_src = read_rotate_half(model, source)
        emb0 = rmod.get_embed(shape=(H,W))
        n_planes=int(emb0.shape[-1]//4)
        entry.update(rope_module=rname, rotate_half=bool(rotate_half),
                     rotate_half_source=rh_src, n_planes=n_planes, rope_rows=int(emb0.shape[0]))
        print(f"  rope='{rname}' planes={n_planes} rotate_half={rotate_half} rows={emb0.shape[0]}")
        gate("rope tensor row count equals patch-token count", label,
             int(emb0.shape[0])==T, f"{emb0.shape[0]} vs {T}")
        amap,row_planes,col_planes,dec = derive_axis_map(emb0,n_planes,rotate_half,H,W)
        freq = freq_strength(emb0,n_planes,rotate_half,H,W)
        entry["axis_map"]=amap; entry["row_planes"]=row_planes; entry["col_planes"]=col_planes
        entry["freq_strength"]={str(k):v for k,v in freq.items()}
        print(f"  axis split: {len(row_planes)} row / {len(col_planes)} col")
        gate("every plane classifies decisively as row or col", label, dec,
             f"max ratio {max(a['ratio'] for a in amap):.2e}")
        gate("axis split is balanced", label,
             len(row_planes)==len(col_planes)==n_planes//2, f"{len(row_planes)}/{len(col_planes)}")

    conds, strat = build_conditions(cfg["family"], n_planes, row_planes, col_planes, freq)
    if strat:
        entry["stratified_control"]=strat
        rs=sorted(row_planes,key=lambda p:-freq[p]); cs=sorted(col_planes,key=lambda p:-freq[p])
        killed=set(strat["planes"])
        row_bands=sorted(i for i,p in enumerate(rs) if p in killed)
        col_bands=sorted(i for i,p in enumerate(cs) if p in killed)
        gate("control removes the SAME frequency bands from both axes", label,
             row_bands==col_bands==sorted(strat["bands"]),
             f"row bands {row_bands} vs col bands {col_bands}")
        rk={p:i for i,p in enumerate(sorted(freq,key=lambda q:-freq[q]))}
        mr=np.mean([rk[p] for p in strat["planes"] if p in row_planes])
        mc=np.mean([rk[p] for p in strat["planes"] if p in col_planes])
        entry["control_global_rank_skew"]=float(abs(mr-mc))
        gate("control global-rank skew is small (W2 unstratified was 2.4)", label,
             abs(mr-mc)<1.5, f"mean global rank row {mr:.1f} vs col {mc:.1f}, skew {abs(mr-mc):.2f}")
    entry["conditions"]={}
    print(f"  {len(conds)} conditions")

    # ---- per-block vs global equivalence, before anything depends on it -----
    if cfg["family"]=="RoPE":
        try:
            g = measure(model,processor,source,RopeGlobalOverride(model,identity_like),
                        [], n_prefix,H,W,"equiv-global")["ssdc"]
            p = measure(model,processor,source,
                        RopePerBlockOverride(model,source,identity_like,blocks=range(N_BLOCKS)),
                        [], n_prefix,H,W,"equiv-perblock")["ssdc"]
            dmax=max(abs(a-b) for a,b in zip(g,p))
            gate("global override hits its W2 anchor", label,
                 abs(g[PL]-A["floor"])<ANCHOR_TOL, f"{g[PL]:+.6f} vs anchor {A['floor']}")
            gate("per-block over all blocks equals the global override", label, dmax<EQUIV_TOL,
                 f"max |delta| over 12 blocks = {dmax:.2e}")
            entry["equivalence_max_delta"]=float(dmax)
        except Exception as ex:
            gate("per-block over all blocks equals the global override", label, False,
                 f"{type(ex).__name__}: {ex}")

    # ---- condition sweep ----------------------------------------------------
    for spec in conds:
        ctx = make_ctx(model, source, spec, cfg["family"], n_planes, rotate_half)
        try:
            r = measure(model,processor,source,ctx,spec["probe_blocks"],n_prefix,H,W,spec["name"])
            r.update({k:spec[k] for k in ("kind","window","read_from","probe_blocks")})
            r["planes_killed"] = ([] if spec["planes"] is None or n_planes is None else
                                  (list(range(n_planes)) if spec["planes"]=="ALL" else list(spec["planes"])))
            entry["conditions"][spec["name"]]=r
            print(f"    {spec['name']:14s} ssdc@{PL}={r['ssdc'][PL]:+.4f}  {r['runtime_s']}s")
        except Exception as ex:
            print(f"    {spec['name']:14s} FAILED {type(ex).__name__}: {ex}")
            entry["conditions"][spec["name"]]={"status":"failed","error":str(ex)}

    # ---- anchors and structural gates ---------------------------------------
    B=entry["conditions"].get("baseline",{}); Fl=entry["conditions"].get("floor",{})
    if "ssdc" in B and "ssdc" in Fl:
        b,f = B["ssdc"], Fl["ssdc"]
        entry["baseline_ssdc"], entry["floor_ssdc"] = b, f
        entry["achievable_range"] = b[PL]-f[PL]
        gate("baseline hits its W2 anchor", label, abs(b[PL]-A["baseline"])<ANCHOR_TOL,
             f"{b[PL]:+.6f} vs anchor {A['baseline']}")
        gate("floor hits its W2 anchor", label, abs(f[PL]-A["floor"])<ANCHOR_TOL,
             f"{f[PL]:+.6f} vs anchor {A['floor']}")
        gate("floor is below baseline", label, f[PL]<b[PL]-0.01, f"{b[PL]:+.4f} -> {f[PL]:+.4f}")
        if cfg["family"]=="RoPE":
            gate("floor equals baseline at block 0", label, abs(b[0]-f[0])<1e-9,
                 f"delta {abs(b[0]-f[0]):.2e}")
        pb=B.get("probes",{}).get(str(PL))
        if pb:
            gate("baseline exact-probe hits its W2 anchor", label,
                 abs(pb["exact"]-A["probe_exact"])<0.03,
                 f"{pb['exact']*100:.2f}% vs anchor {A['probe_exact']*100:.2f}%")

        # prefix conditions only: the window must sit at the floor
        for nm,r in entry["conditions"].items():
            if r.get("kind")!="prefix" or "ssdc" not in r: continue
            lb=max(r["window"])+1
            ok=abs(r["ssdc"][lb]-f[lb])<FLOOR_TOL
            gate(f"{nm}: window sits at the floor (block {lb})", label, ok,
                 f"{r['ssdc'][lb]:+.6f} vs floor {f[lb]:+.6f}")

        # axis conditions: expectation is on the probes, never on in-window SSDC
        for nm,r in entry["conditions"].items():
            if r.get("kind")!="axis" or "probes" not in r: continue
            masked_axis = "row" if nm.startswith("axisrow") else "col"
            surv = "col" if masked_axis=="row" else "row"
            lb=max(r["window"])+1
            p=r["probes"].get(str(lb)); pb0=B["probes"].get(str(lb))
            if not p or not pb0: continue
            ch=p["chance"][masked_axis]
            gate(f"{nm}: masked {masked_axis} probe at chance in-window (block {lb})", label,
                 abs(p[masked_axis]-ch)<PROBE_CHANCE_TOL,
                 f"{p[masked_axis]*100:.1f}% vs chance {ch*100:.1f}%")
            gate(f"{nm}: surviving {surv} probe near baseline in-window (block {lb})", label,
                 abs(p[surv]-pb0[surv])<PROBE_BASE_TOL,
                 f"{p[surv]*100:.1f}% vs baseline {pb0[surv]*100:.1f}%")
            # a masked axis must stay at chance at EVERY in-window block
            bad=[bb for bb in range(1,lb+1)
                 if str(bb) in r["probes"]
                 and r["probes"][str(bb)][masked_axis]-r["probes"][str(bb)]["chance"][masked_axis] > PROBE_CHANCE_TOL]
            gate(f"{nm}: masked axis stays at chance across the whole window", label, not bad,
                 f"blocks above chance: {bad}" if bad else "")

        # random control: below baseline, above chance. No tighter expectation.
        for nm,r in entry["conditions"].items():
            if r.get("kind")!="rand" or "probes" not in r: continue
            lb=max(r["window"])+1
            p=r["probes"].get(str(lb)); pb0=B["probes"].get(str(lb))
            if not p or not pb0: continue
            ok=all(p["chance"][h] < p[h] + 1e-9 and p[h] <= pb0[h]+PROBE_CHANCE_TOL for h in ("row","col"))
            gate(f"{nm}: probes between chance and baseline in-window", label, ok,
                 f"row {p['row']*100:.1f}% col {p['col']*100:.1f}% vs baseline "
                 f"{pb0['row']*100:.1f}/{pb0['col']*100:.1f}%")

    entry["runtime_s"]=round(time.time()-t_model,1)
    RESULTS[label]=entry
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()
    print(f"  model done in {entry['runtime_s']}s")

# ---- cell 21 ----
import pandas as pd
pd.set_option("display.width",240); pd.set_option("display.max_columns",60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

def ssdc_ret(e,r,b): 
    rng=e["baseline_ssdc"][b]-e["floor_ssdc"][b]
    return (r["ssdc"][b]-e["floor_ssdc"][b])/rng if rng>1e-6 else float("nan")

def probe_ret(e,r,b,h):
    pb=e["conditions"]["baseline"]["probes"].get(str(b)); p=r.get("probes",{}).get(str(b))
    if not pb or not p: return float("nan")
    ch=p["chance"][h]; rng=pb[h]-ch
    return (p[h]-ch)/rng if rng>1e-6 else float("nan")

rows=[]
for lab,e in RESULTS.items():
    if e.get("status")!="ok" or "baseline_ssdc" not in e: continue
    for nm,r in e["conditions"].items():
        if "ssdc" not in r or nm in ("baseline","floor"): continue
        for b in range(r.get("read_from",0), N_BLOCKS):
            rows.append({"run":lab,"condition":nm,"kind":r["kind"],"block":b,
                         "ssdc":r["ssdc"][b],"ssdc_ret":ssdc_ret(e,r,b),
                         "exact_ret":probe_ret(e,r,b,"exact"),
                         "row_ret":probe_ret(e,r,b,"row"),
                         "col_ret":probe_ret(e,r,b,"col")})
curve=pd.DataFrame(rows)
print("RECOVERY CURVES: retention by block\n")
for lab in RESULTS:
    sub=curve[(curve.run==lab)&(curve.kind.isin(["prefix","middle"]))]
    if sub.empty: continue
    print(f"--- {lab}")
    print(sub.pivot_table(index="condition",columns="block",values="ssdc_ret").round(3).to_string())
    print()

# ---- cell 22 ----
# half-recovery and final retention per condition
def half_recovery(vals):
    '''Blocks after the first readout block until SSDC retention first reaches 0.5.'''
    for i,v in enumerate(vals):
        if np.isfinite(v) and v>=0.5:
            if i==0: return 0.0
            prev=vals[i-1]
            if not np.isfinite(prev) or v==prev: return float(i)
            return float(i-1+(0.5-prev)/(v-prev))
    return None

summ=[]
for lab,e in RESULTS.items():
    if e.get("status")!="ok" or "baseline_ssdc" not in e: continue
    sd=NOISE_SD.get(lab); PL=e["probe_layer"]
    for nm,r in e["conditions"].items():
        if "ssdc" not in r or nm in ("baseline","floor"): continue
        blocks=list(range(r["read_from"],N_BLOCKS))
        vals=[ssdc_ret(e,r,b) for b in blocks]
        onset=None
        if sd:
            for b,v in zip(blocks,vals):
                rng=e["baseline_ssdc"][b]-e["floor_ssdc"][b]
                if np.isfinite(v) and v*rng > ONSET_K*sd: onset=b; break
        summ.append({"run":lab,"condition":nm,"kind":r["kind"],
                     "window":f"{min(r['window'])}-{max(r['window'])}" if r["window"] else "-",
                     "read_from":r["read_from"],
                     "half_recovery_blocks":half_recovery(vals),
                     "final_ssdc_ret":vals[-1] if vals else np.nan,
                     "final_exact_ret":probe_ret(e,r,N_BLOCKS-1,"exact"),
                     "onset_block":onset})
summary=pd.DataFrame(summ)
print("CONDITION SUMMARY\n")
for lab in RESULTS:
    s=summary[summary.run==lab]
    if s.empty: continue
    print(f"--- {lab}"); print(s.drop(columns=["run"]).to_string(index=False)); print()

# ---- cell 23 ----
# The axis-regeneration readout. k=5 is the comparison anchor; the other row windows
# are there for dose dependence and have no matching column curve by design.
print("AXIS REGENERATION: each head against its own baseline\n")
for lab,e in RESULTS.items():
    if e.get("family")!="RoPE" or e.get("status")!="ok": continue
    print(f"--- {lab}")
    for nm in [f"axisrow_{k}" for k in AXISROW_KS]+[f"axiscol_{k}" for k in AXISCOL_KS]+[f"rand16_{k}" for k in RAND_KS]:
        r=e["conditions"].get(nm)
        if not r or "ssdc" not in r: continue
        lb=max(r["window"])+1
        print(f"  {nm}  (masked 0-{max(r['window'])}, recovery from block {r['read_from']})")
        print(f"    {'blk':>3} {'ssdc_ret':>9} {'row_ret':>8} {'col_ret':>8} {'exact_ret':>10}")
        for b in range(1,N_BLOCKS):
            if str(b) not in r.get("probes",{}): continue
            tag="  in-window" if b<=lb else ""
            print(f"    {b:3d} {ssdc_ret(e,r,b):9.3f} {probe_ret(e,r,b,'row'):8.3f} "
                  f"{probe_ret(e,r,b,'col'):8.3f} {probe_ret(e,r,b,'exact'):10.3f}{tag}")
        print()
print("Reading: axis-specific regeneration means the masked head's retention climbs after the")
print("window while the surviving head stays near 1.0, AND rand16 recovers less than axisrow_5.")
print()
print("NOTE: in-window ssdc_ret is NOT meaningful for axis conditions and may exceed 1.0.")
print("The surviving axis keeps tracking half the L1 distance, and at early blocks the")
print("baseline-minus-floor range is small, so the ratio is unstable. In-window the probes")
print("are the readout; SSDC only becomes interpretable from read_from onward.")

# ---- cell 24 ----
ok=[(l,e) for l,e in RESULTS.items() if e.get("status")=="ok" and "baseline_ssdc" in e and e["family"]=="RoPE"]
if ok:
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(ok),figsize=(5.2*len(ok),4.0),sharey=True)
    axes=np.atleast_1d(axes); cm=plt.get_cmap("viridis")
    for ax,(lab,e) in zip(axes,ok):
        pref=[n for n in e["conditions"] if n.startswith("prefix_")]
        pref=sorted(pref,key=lambda s:int(s.split("_")[1]))
        for i,nm in enumerate(pref):
            r=e["conditions"][nm]
            if "ssdc" not in r: continue
            bs=list(range(r["read_from"],N_BLOCKS))
            ax.plot(bs,[ssdc_ret(e,r,b) for b in bs],"o-",ms=3,lw=1.5,
                    color=cm(i/max(len(pref)-1,1)),label=nm)
        ax.axhline(1.0,color="k",lw=0.8,alpha=0.5); ax.axhline(0,color="k",lw=0.5,alpha=0.3)
        ax.set_title(lab,fontsize=9); ax.set_xlabel("block"); ax.grid(alpha=0.25)
        ax.legend(fontsize=6,ncol=2)
    axes[0].set_ylabel("SSDC retention")
    fig.suptitle("Re-injection after the masking window",y=1.02); fig.tight_layout(); plt.show()

# ---- cell 26 ----
gdf=pd.DataFrame(GATES)
print("GATE REPORT\n")
print(gdf.to_string(index=False) if len(gdf) else "(none)")
n_fail=int((gdf.status=="FAIL").sum()) if len(gdf) else 0
print("\n"+"="*78)
print("ALL GATES PASSED" if n_fail==0 else f"{n_fail} GATE(S) FAILED: report before using")
if n_fail:
    for _,r in gdf[gdf.status=="FAIL"].iterrows():
        print(f"   - [{r['scope']}] {r['gate']}: {r['detail']}")
print("="*78)

# ---- cell 28 ----
export={
 "notebook":"W3","generated_utc":RUN_STAMP,
 "environment":{"torch":torch.__version__,"timm":timm.__version__,"device":DEVICE,
                "dtype":"float32","repo_path":str(repo),
                "gpu":torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"},
 "measurement":{"implementation":"repo metrics/ssdc.py evaluate_ssdc",
                "RPI":True,"number_images":NUMBER_IMAGES,"batch_size":BATCH_SIZE,
                "metric":METRIC,"num_workers":0,"torch_seed":TORCH_SEED,
                "mask_seed":MASK_SEED,"input_size":INPUT_SIZE,
                "precision":"fp32; baseline and floor re-measured in-run"},
 "probe_settings":{"lr":PROBE_LR,"passes":PROBE_PASSES,"batch":PROBE_BATCH,
                   "train_frac":PROBE_TRAIN_FRAC,"split":"by image","standardised":True,
                   "heads":["exact","row","col"],"reported":"peak test accuracy over passes",
                   "device":PDEV},
 "comparison_policy":"intra-model only",
 "design_notes":{"readout_offset":"first recoverable block is last_masked + 2",
                 "axis_anchor":"k=5 is the row-vs-column comparison point; axisrow_2 and "
                               "axisrow_8 are for dose dependence and have no column counterpart",
                 "floor_gate_scope":"prefix conditions only; axis conditions gate on probes"},
 "w2_anchors":W2_ANCHORS,
 "results":RESULTS,"gates":GATES,
 "gate_summary":{"total":len(GATES),"pass":int(sum(g["status"]=="PASS" for g in GATES)),"fail":n_fail},
 "normalisation_protocol":{
   "formula":"(observed_b - floor_b) / (baseline_b - floor_b), per block",
   "source":"in-run fp32 baseline and floor from this notebook",
   "floors":{l:(e["floor_ssdc"] if e.get("status")=="ok" and "floor_ssdc" in e else None)
             for l,e in RESULTS.items()}},
}
out_json=OUT_DIR/f"w3_windows_{RUN_STAMP.replace(':','-')}.json"
out_json.write_text(json.dumps(export,indent=2))
curve.to_csv(OUT_DIR/"w3_curves.csv",index=False)
summary.to_csv(OUT_DIR/"w3_summary.csv",index=False)
gdf.to_csv(OUT_DIR/"w3_gate_report.csv",index=False)

pb=[]
for lab,e in RESULTS.items():
    if e.get("status")!="ok": continue
    for nm,r in e.get("conditions",{}).items():
        if "ssdc" not in r: continue
        for b,v in enumerate(r["ssdc"]): pb.append({"run":lab,"condition":nm,"block":b,"ssdc":v})
pd.DataFrame(pb).to_csv(OUT_DIR/"w3_per_block.csv",index=False)

import shutil
archive=shutil.make_archive("w3_output","zip",OUT_DIR)
print("written:")
for p in sorted(OUT_DIR.iterdir()): print(f"  {p}  ({p.stat().st_size:,} bytes)")
print(f"\narchive: {archive} ({os.path.getsize(archive):,} bytes)")
try:
    from google.colab import files as cf; cf.download(archive)
except Exception:
    print("(not Colab: copy the archive back manually)")
