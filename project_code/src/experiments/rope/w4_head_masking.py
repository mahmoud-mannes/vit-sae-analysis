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
HEAD_SEED     = 4127          # random-head control draws
INPUT_SIZE    = 224

PROBE_LR, PROBE_PASSES, PROBE_BATCH, PROBE_TRAIN_FRAC = 5e-3, 20, 512, 0.8
PROBE_CHUNK   = 8192
PROBE_DEVICE  = None          # None = model device; "cpu" to force probes off the GPU

ANCHOR_TOL      = 0.01        # |measured - W3 anchor|
REPRO_TOL       = 1e-6        # tolerance for re-running W3 conditions
EQUIV_TOL       = 1e-9
CHANCE_TOL_RET  = 0.05        # probe bands are in RETENTION units, not raw accuracy

WINDOW        = list(range(0, 6))    # blocks 0-5, the window W3 characterised on both models
POST          = list(range(6, 12))   # where a knocked-out head could do re-injection
BK            = [1, 2, 4, 8]         # cumulative top-k for stage B
BCTL_K        = [2, 4, 8]            # random-head control sizes
D_TOP_N       = 2                    # how many stage-C heads get the axis treatment

# --- selection rules, fixed before the run ---------------------------------
RANK_A = "lowest SSDC retention at the probe layer (all-blocks single-head mask); ties by head index"
RANK_C = "largest drop in block-11 SSDC retention vs the prefix_5 anchor; ties by head index"

# --- probe depth per stage -------------------------------------------------
PROBE_DEPTH = {"baseline":"all", "floor":"all",
               "A":"probe_layer", "B":"sparse", "Bctl":"sparse",
               "Cref":"post", "C":"post", "Dref":"post", "D":"post"}

# W3 fp32 values, used as absolute anchors
W3_ANCHORS = {
    "RoPE · naver":  dict(probe_layer=4, n_heads=12, baseline=0.469422, floor=0.012865,
                          prefix_5_b11=0.329663, axisrow_5_b11=0.382360, axiscol_5_b11=0.419516,
                          probe_exact=0.3297, probe_row=0.6520, probe_col=0.6283),
    "RoPE · DINOv3": dict(probe_layer=4, n_heads=12, baseline=0.948654, floor=0.017712,
                          prefix_5_b11=0.442640, axisrow_5_b11=0.627570, axiscol_5_b11=0.652320,
                          probe_exact=0.3960, probe_row=0.7560, probe_col=0.7643),
}

MODELS = {
    "RoPE · naver":  dict(loader="load_rope", kwargs=dict(input_size=INPUT_SIZE)),
    "RoPE · DINOv3": dict(loader="load_rope",
                          kwargs=dict(model_name="vit_base_patch16_dinov3.lvd1689m",
                                      input_size=INPUT_SIZE)),
}

OUT_DIR = Path("w4_output"); OUT_DIR.mkdir(exist_ok=True)
# ============================================================================

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDEV = PROBE_DEVICE or DEVICE
N_BLOCKS = 12

GATES=[]
def gate(name, scope, passed, detail=""):
    GATES.append({"gate":name,"scope":scope,"status":"PASS" if passed else "FAIL","detail":detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {scope}: {name}" + (f": {detail}" if detail else ""))
    return passed

print(f"device {DEVICE} | probes on {PDEV} | fp32 | images {NUMBER_IMAGES} | batch {BATCH_SIZE}")
print(f"window {WINDOW[0]}-{WINDOW[-1]}  post {POST[0]}-{POST[-1]}")
print(f"RANK_A: {RANK_A}")
print(f"RANK_C: {RANK_C}")
print(f"run {RUN_STAMP}")

# ---- cell 7 ----
def resolve_repo_path():
    cands=[]
    if REPO_PATH: cands.append(Path(REPO_PATH).expanduser().resolve())
    clone=Path("vit-sae-analysis")
    cands += [clone/"project_code"/"src", clone/"src", clone]
    for c in cands:
        if (c/"metrics").exists() and (c/"main").exists(): return c
    if REPO_URL and not clone.exists():
        print(f"cloning {REPO_URL} ...")
        subprocess.check_call(["git","clone","--depth","1",REPO_URL,str(clone)])
        for c in cands:
            if (c/"metrics").exists() and (c/"main").exists(): return c
    raise FileNotFoundError("No checkout with metrics/ and main/. Tried: "+", ".join(str(c) for c in cands))

repo=resolve_repo_path()
if str(repo) not in sys.path: sys.path.insert(0,str(repo))
print("repo root:",repo)

from metrics.ssdc import evaluate_ssdc, spatial_similarity_distance_correlation
from main.load_models import (get_vit_blocks, get_block_attention, get_patch_embed_conv,
                              load_rope)
from main.model import predict
from main.prep_data import prep_data
from experiments.common import load_imagenet
LOADERS={"load_rope":load_rope}
print("imported evaluate_ssdc, load_rope, predict, prep_data, load_imagenet")

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

DATASET=load_imagenet(split="validation", streaming=True)
print("dataset ready:",type(DATASET).__name__)

# ---- cell 9 ----
def find_rope_module(model):
    for name,mod in model.named_modules():
        if callable(getattr(mod,"get_embed",None)): return name,mod
    return None,None

def capture_forward_grid(model, processor, source):
    grid={}
    conv=get_patch_embed_conv(model,source)
    h=conv.register_forward_hook(
        lambda m,i,o: grid.update(hw=(int(o.shape[2]),int(o.shape[3])) if o.dim()==4 else None))
    dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=BATCH_SIZE,
                 batch_size=BATCH_SIZE,half=False,num_workers=0)
    with torch.no_grad():
        for images,_ in dl:
            model(images.to(DEVICE)); break
    h.remove()
    return grid.get("hw")

def read_attn_attrs(model, source):
    '''rotate_half, num_heads and num_prefix_tokens, read off the attention module.'''
    rh, rh_src, nh, npt = False, "default(interleaved)", None, None
    for blk in get_vit_blocks(model, source):
        attn=get_block_attention(blk, source)
        for attr in ("rotate_half","half","rope_rotate_half"):
            v=getattr(attn,attr,None)
            if isinstance(v,bool): rh,rh_src=v,attr; break
        nh=getattr(attn,"num_heads",None)
        npt=getattr(attn,"num_prefix_tokens",None)
        break
    return rh, rh_src, nh, npt

def plane_columns(p,n_planes,rotate_half):
    return [p,p+n_planes] if rotate_half else [2*p,2*p+1]

def derive_axis_map(emb,n_planes,rotate_half,H,W):
    sin,cos=emb.detach().float().chunk(2,-1)
    rows,dec=[],True
    for p in range(n_planes):
        c=plane_columns(p,n_planes,rotate_half)[0]
        z=torch.complex(cos[:,c],sin[:,c]).reshape(H,W)
        ra=(z-z.mean(dim=1,keepdim=True)).abs().mean().item()
        rb=(z-z.mean(dim=0,keepdim=True)).abs().mean().item()
        axis="row" if ra<rb else "col"
        lo,hi=sorted([ra,rb]); ratio=lo/hi if hi>1e-12 else 1.0
        if ratio>0.10: dec=False
        rows.append({"plane":p,"axis":axis,"ratio":ratio})
    rp=[r["plane"] for r in rows if r["axis"]=="row"]
    cp=[r["plane"] for r in rows if r["axis"]=="col"]
    return rows,rp,cp,dec

# ---- cell 11 ----
def mask_planes(emb,kill,n_planes,rotate_half):
    if not kill: return emb
    sin,cos=emb.chunk(2,-1); sin,cos=sin.clone(),cos.clone()
    for p in kill:
        for c in plane_columns(p,n_planes,rotate_half):
            sin[...,c]=0.0; cos[...,c]=1.0
    return torch.cat([sin,cos],-1)

def identity_like(emb):
    sin,cos=emb.chunk(2,-1)
    return torch.cat([torch.zeros_like(sin),torch.ones_like(cos)],-1)

def per_head(heads, inner, n_heads):
    '''Return a transform that expands rope to (H,N,2D) and applies `inner` to `heads` only.

    emb is only ever chunk()ed and broadcast-multiplied inside apply_rot_embed_cat, so a
    leading head dimension broadcasts correctly against q of shape (B,H,N,D).
    '''
    hs=sorted(set(heads))
    def tf(emb):
        if emb.dim()!=2:
            raise RuntimeError(f"per_head expected a 2-D rope tensor, got {tuple(emb.shape)} "
                               ": two overrides are stacking on the same block")
        out=emb.unsqueeze(0).repeat(n_heads,1,1)
        if hs:
            t=inner(emb)
            for h in hs: out[h]=t
        return out
    return tf

class RopeGlobalOverride:
    def __init__(self,model,transform):
        self.name,self.mod=find_rope_module(model); self.transform=transform; self.orig=None
        self.handles=[]
    def __enter__(self):
        if self.mod is None: raise RuntimeError("no get_embed module")
        self.orig=self.mod.get_embed
        o,tf=self.orig,self.transform
        self.mod.get_embed=(lambda _o=o,_tf=tf:(lambda *a,**k:_tf(_o(*a,**k))))()
        self.handles=[1]     # so hooks_registered is non-zero for this mechanism too
        return self
    def __exit__(self,*e):
        self.mod.get_embed=self.orig; self.handles=[]

class RopeComposite:
    '''Apply per-segment transforms to disjoint block ranges via forward pre-hooks.'''
    def __init__(self, model, source, segments):
        self.blocks=get_vit_blocks(model,source); self.source=source
        self.segments=[(list(b),t) for b,t in segments]
        seen=set()
        for b,_ in self.segments:
            if seen & set(b):
                raise ValueError(f"overlapping segments: {sorted(seen & set(b))}")
            seen |= set(b)
        self.handles=[]
    def __enter__(self):
        def make(tf):
            def pre(module,args,kwargs):
                r=kwargs.get("rope",None)
                if r is not None:
                    kwargs=dict(kwargs); kwargs["rope"]=tf(r)
                return args,kwargs
            return pre
        for blocks,tf in self.segments:
            for i in blocks:
                attn=get_block_attention(self.blocks[i],self.source)
                self.handles.append(attn.register_forward_pre_hook(make(tf),with_kwargs=True))
        return self
    def __exit__(self,*e):
        for h in self.handles: h.remove()
        self.handles=[]

class NullCtx:
    handles=[]
    def __enter__(self): return self
    def __exit__(self,*e): return False

# ---- cell 13 ----
def _rot_il(x):
    x=x.unflatten(-1,(-1,2)); return torch.stack([-x[...,1],x[...,0]],-1).flatten(-2)
def _rot_half(x):
    a,b=x.chunk(2,-1); return torch.cat([-b,a],-1)
def _apply(x,emb,half=False):
    s,c=emb.chunk(2,-1)
    return x*c + (_rot_half(x) if half else _rot_il(x))*s

for lname,rh in [("interleaved (naver)",False),("rotate_half (DINOv3)",True)]:
    npl,dim,H=32,64,12
    e=torch.randn(196,2*dim); x=torch.randn(2,H,196,dim)
    gate(f"empty plane mask is bit-identical [{lname}]","algebra",
         torch.equal(mask_planes(e,[],npl,rh),e))
    gate(f"full plane mask equals identity [{lname}]","algebra",
         torch.allclose(mask_planes(e,list(range(npl)),npl,rh),identity_like(e)))
    m=mask_planes(e,[3],npl,rh)
    touched=torch.nonzero((m-e).abs().sum(0)>0).flatten().tolist()
    sc=plane_columns(3,npl,rh); expect=sorted(set(sc+[c+dim for c in sc]))
    gate(f"masking plane 3 touches only its own columns [{lname}]","algebra",
         touched==expect,f"{touched} vs {expect}")

    # --- per-head broadcast ------------------------------------------------
    full=_apply(x,e,rh); iden=_apply(x,identity_like(e),rh)
    none_=_apply(x,per_head([],identity_like,H)(e),rh)
    all_ =_apply(x,per_head(range(H),identity_like,H)(e),rh)
    one  =_apply(x,per_head([3],identity_like,H)(e),rh)
    others=[h for h in range(H) if h!=3]
    gate(f"per-head tensor has shape (H,N,2D) [{lname}]","per-head",
         tuple(per_head([3],identity_like,H)(e).shape)==(H,196,2*dim),
         str(tuple(per_head([3],identity_like,H)(e).shape)))
    gate(f"masking NO heads equals unmodified rope [{lname}]","per-head",
         torch.allclose(none_,full,atol=1e-6))
    gate(f"masking ALL heads equals global identity [{lname}]","per-head",
         torch.allclose(all_,iden,atol=1e-6))
    gate(f"masking head 3 leaves other heads untouched [{lname}]","per-head",
         torch.allclose(one[:,others],full[:,others],atol=1e-6))
    gate(f"masking head 3 actually changes head 3 [{lname}]","per-head",
         not torch.allclose(one[:,3],full[:,3]))

class _StubModel:
    def __init__(self,n=12): self.blocks=[nn.Identity() for _ in range(n)]

_stub=_StubModel()
try:
    RopeComposite(_stub,"timm",[(range(0,6),identity_like),(range(5,12),identity_like)])
    _overlap_caught=False
except ValueError as ex:
    _overlap_caught="overlapping" in str(ex)
gate("composite REJECTS overlapping segments","composite",_overlap_caught,
     "constructing 0-5 with 5-11 must raise")
try:
    RopeComposite(_stub,"timm",[(range(0,6),identity_like),(range(6,12),identity_like)])
    _disjoint_ok=True
except Exception as ex:
    _disjoint_ok=False
gate("composite ACCEPTS disjoint segments","composite",_disjoint_ok,"0-5 with 6-11 must construct")

# ---- cell 15 ----
class Capture:
    def __init__(self,model,source,blocks,n_prefix):
        self.blocks=get_vit_blocks(model,source); self.which=list(blocks)
        self.n_prefix=int(n_prefix); self.source=source
        self.store={i:[] for i in self.which}; self.handles=[]
    def __enter__(self):
        def make(i):
            def hook(module,inputs,output):
                t=inputs[0].detach()
                if self.n_prefix>0: t=t[:,self.n_prefix:,:]
                self.store[i].append(t.float().cpu())
            return hook
        for i in self.which:
            attn=get_block_attention(self.blocks[i],self.source)
            self.handles.append(attn.register_forward_hook(make(i)))
        return self
    def __exit__(self,*e):
        for h in self.handles: h.remove()
        self.handles=[]
    def stacked(self,i):
        return torch.cat(self.store[i],0) if self.store[i] else None

def _fit_head(Xtr,ytr,Xte,yte,n_classes):
    head=nn.Linear(Xtr.shape[1],n_classes).to(Xtr.device)
    opt=torch.optim.AdamW(head.parameters(),lr=PROBE_LR)
    n=Xtr.shape[0]; peak=0.0
    for _ in range(PROBE_PASSES):
        perm=torch.randperm(n,device=Xtr.device); head.train()
        for s in range(0,n,PROBE_BATCH):
            b=perm[s:s+PROBE_BATCH]
            opt.zero_grad(); F.cross_entropy(head(Xtr[b]),ytr[b]).backward(); opt.step()
        head.eval(); correct=0
        with torch.no_grad():
            for s in range(0,Xte.shape[0],PROBE_CHUNK):
                correct+=(head(Xte[s:s+PROBE_CHUNK]).argmax(-1)==yte[s:s+PROBE_CHUNK]).sum().item()
        peak=max(peak,correct/Xte.shape[0])
    del head,opt
    return float(peak)

def probe_block(feats,H,W,n_images,seed=TORCH_SEED):
    T=H*W
    g=torch.Generator().manual_seed(seed)
    idx=torch.randperm(n_images,generator=g)
    n_tr=int(round(n_images*PROBE_TRAIN_FRAC))
    def rows(ids): return (ids.unsqueeze(1)*T+torch.arange(T).unsqueeze(0)).reshape(-1)
    tr,te=rows(idx[:n_tr]),rows(idx[n_tr:])
    Xtr=feats[tr].to(PDEV); Xte=feats[te].to(PDEV)
    mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True).clamp_min(1e-6)
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    slot=torch.arange(T); out={}
    for name,lab,nc in (("exact",slot,T),("row",slot//W,H),("col",slot%W,W)):
        y=lab.repeat(n_images)
        out[name]=_fit_head(Xtr,y[tr].to(PDEV),Xte,y[te].to(PDEV),nc)
    out["chance"]={"exact":1.0/T,"row":1.0/H,"col":1.0/W}
    del Xtr,Xte
    if PDEV=="cuda": torch.cuda.empty_cache()
    return out

# ---- cell 17 ----
def measure(model, processor, source, ctx, probe_blocks, n_prefix, H, W):
    torch.manual_seed(TORCH_SEED); np.random.seed(TORCH_SEED%(2**31))
    t0=time.time()
    cap=Capture(model,source,probe_blocks,n_prefix) if probe_blocks else None
    if cap: cap.__enter__()
    hooks=0
    try:
        with ctx:
            hooks=len(getattr(ctx,"handles",[]) or [])
            scores,_=evaluate_ssdc(model,processor,DATASET,source,RPI=True,
                                   number_images=NUMBER_IMAGES,batch_size=BATCH_SIZE,
                                   metric=METRIC,half=False,num_workers=0)
    finally:
        if cap: cap.__exit__()
    res={"ssdc":[float(s) for s in scores],"probes":{},"hooks_registered":hooks}
    if cap:
        for i in probe_blocks:
            acts=cap.stacked(i)
            if acts is None: continue
            res["probes"][str(i)]=probe_block(acts.reshape(-1,acts.shape[-1]),H,W,acts.shape[0])
            del acts
        cap.store.clear()
    res["runtime_s"]=round(time.time()-t0,1)
    return res

def depth_blocks(kind, probe_layer):
    d=PROBE_DEPTH[kind]
    if d=="all":         return list(range(N_BLOCKS))
    if d=="probe_layer": return [probe_layer]
    if d=="sparse":      return sorted({probe_layer,min(probe_layer+2,N_BLOCKS-1),
                                        min(probe_layer+4,N_BLOCKS-1),N_BLOCKS-1})
    if d=="post":        return [b for b in range(POST[0]+1, N_BLOCKS)]
    raise ValueError(d)

# ---- cell 19 ----
def sret(entry,ssdc,b):
    rng=entry["baseline_ssdc"][b]-entry["floor_ssdc"][b]
    return (ssdc[b]-entry["floor_ssdc"][b])/rng if rng>1e-6 else float("nan")

RESULTS={}

for label,cfg in MODELS.items():
    print("\n"+"="*78); print(label); print("="*78)
    t_model=time.time()
    A=W3_ANCHORS[label]; PL=A["probe_layer"]

    model,processor,source=LOADERS[cfg["loader"]](device=DEVICE,half=False,**cfg["kwargs"])
    model.eval().float()

    rotate_half,rh_src,n_heads_attr,npt_attr=read_attn_attrs(model,source)
    n_prefix=int(getattr(model,"num_prefix_tokens",1))
    hw=capture_forward_grid(model,processor,source)
    if hw is None: raise RuntimeError("could not read the forward-pass grid")
    Hg,Wg=hw; T=Hg*Wg
    n_heads=int(n_heads_attr) if n_heads_attr else A["n_heads"]

    rname,rmod=find_rope_module(model)
    emb0=rmod.get_embed(shape=(Hg,Wg))
    n_planes=int(emb0.shape[-1]//4)
    amap,row_planes,col_planes,dec=derive_axis_map(emb0,n_planes,rotate_half,Hg,Wg)

    entry={"status":"ok","source":source,"probe_layer":PL,"n_blocks":len(get_vit_blocks(model,source)),
           "num_prefix_tokens":n_prefix,"grid":[Hg,Wg],"patch_tokens":T,"dtype":"float32",
           "rope_module":rname,"rotate_half":bool(rotate_half),"rotate_half_source":rh_src,
           "n_planes":n_planes,"n_heads":n_heads,"row_planes":row_planes,"col_planes":col_planes,
           "axis_map":amap,"window":WINDOW,"post":POST,
           "rank_rule_A":RANK_A,"rank_rule_C":RANK_C,"conditions":{}}
    print(f"  source={source} heads={n_heads} prefix={n_prefix} grid={Hg}x{Wg} "
          f"planes={n_planes} rotate_half={rotate_half}")
    gate("forward grid is 14x14",label,(Hg,Wg)==(14,14),f"{Hg}x{Wg}")
    gate("rope rows equal patch tokens",label,int(emb0.shape[0])==T,f"{emb0.shape[0]} vs {T}")
    gate("num_heads matches the anchor",label,n_heads==A["n_heads"],f"{n_heads} vs {A['n_heads']}")
    gate("every plane classifies decisively",label,dec,f"max ratio {max(a['ratio'] for a in amap):.1e}")
    gate("axis split is balanced",label,len(row_planes)==len(col_planes)==n_planes//2,
         f"{len(row_planes)}/{len(col_planes)}")

    IDEN=identity_like
    ROWM=lambda e: mask_planes(e,row_planes,n_planes,rotate_half)
    COLM=lambda e: mask_planes(e,col_planes,n_planes,rotate_half)

    def run(name,kind,ctx,extra=None):
        pb=depth_blocks(kind,PL)
        r=measure(model,processor,source,ctx,pb,n_prefix,Hg,Wg)
        r["kind"]=kind; r["probe_blocks"]=pb
        if extra: r.update(extra)
        entry["conditions"][name]=r
        tag=f"ssdc@{PL}={r['ssdc'][PL]:+.4f}"
        if 11>=min(pb) or kind in("Cref","C","Dref","D"): tag+=f" b11={r['ssdc'][11]:+.4f}"
        print(f"    {name:16s} {tag}  hooks={r['hooks_registered']:2d}  {r['runtime_s']}s")
        return r

    # ---- baseline / floor ------------------------------------------------
    rb=run("baseline","baseline",NullCtx())
    entry["baseline_ssdc"]=rb["ssdc"]
    rf=run("floor","floor",RopeGlobalOverride(model,IDEN))
    entry["floor_ssdc"]=rf["ssdc"]
    gate("baseline hits its W3 anchor",label,abs(rb["ssdc"][PL]-A["baseline"])<ANCHOR_TOL,
         f"{rb['ssdc'][PL]:+.6f} vs {A['baseline']}")
    gate("floor hits its W3 anchor",label,abs(rf["ssdc"][PL]-A["floor"])<ANCHOR_TOL,
         f"{rf['ssdc'][PL]:+.6f} vs {A['floor']}")
    gate("floor equals baseline at block 0",label,abs(rb["ssdc"][0]-rf["ssdc"][0])<1e-9,
         f"delta {abs(rb['ssdc'][0]-rf['ssdc'][0]):.2e}")

    # ---- mechanism gate: all heads masked must equal the global floor -----
    rall=measure(model,processor,source,
                 RopeComposite(model,source,[(range(N_BLOCKS),per_head(range(n_heads),IDEN,n_heads))]),
                 [],n_prefix,Hg,Wg)
    dmax=max(abs(a-b) for a,b in zip(rall["ssdc"],rf["ssdc"]))
    gate("per-head mask of ALL heads reproduces the global floor",label,dmax<EQUIV_TOL,
         f"max |delta| over 12 blocks = {dmax:.2e}")
    entry["allheads_vs_floor_max_delta"]=float(dmax)

    # ---- STAGE A ---------------------------------------------------------
    print("  stage A: single head, all blocks")
    for h in range(n_heads):
        run(f"A_head{h:02d}","A",
            RopeComposite(model,source,[(range(N_BLOCKS),per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h],"blocks_masked":list(range(N_BLOCKS))})
    rankA=sorted(range(n_heads),
                 key=lambda h:(sret(entry,entry["conditions"][f"A_head{h:02d}"]["ssdc"],PL),h))
    entry["stageA_ranking"]=[{"head":h,
        "ssdc_retention_at_probe":sret(entry,entry["conditions"][f"A_head{h:02d}"]["ssdc"],PL)}
        for h in rankA]
    print("  stage A ranking (most important first):",rankA)
    (OUT_DIR/f"w4_stageA_{label.split('·')[1].strip()}_{RUN_STAMP.replace(':','-')}.json").write_text(
        json.dumps({"ranking":entry["stageA_ranking"],"rule":RANK_A},indent=2))
    gate("stage A ranking is a permutation of all heads",label,
         sorted(rankA)==list(range(n_heads)),f"top: {rankA[:4]}")

    # ---- STAGE B and its random control ----------------------------------
    print("  stage B: cumulative top-k, all blocks")
    for k in BK:
        hs=rankA[:k]
        run(f"B_top{k}","B",
            RopeComposite(model,source,[(range(N_BLOCKS),per_head(hs,IDEN,n_heads))]),
            {"heads_masked":hs,"blocks_masked":list(range(N_BLOCKS))})
    rng_h=np.random.default_rng(HEAD_SEED)
    entry["random_head_draws"]={}
    for k in BCTL_K:
        hs=sorted(rng_h.choice(n_heads,size=k,replace=False).tolist())
        entry["random_head_draws"][str(k)]={"heads":hs,
            "overlap_with_topk":sorted(set(hs)&set(rankA[:k]))}
        run(f"Bctl_rand{k}","Bctl",
            RopeComposite(model,source,[(range(N_BLOCKS),per_head(hs,IDEN,n_heads))]),
            {"heads_masked":hs,"blocks_masked":list(range(N_BLOCKS))})

    # ---- STAGE C: knock a head out of the re-injection segment ------------
    print("  stage C: window 0-5 off, plus one head off in 6-11")
    rcref=run("Cref_prefix5","Cref",
              RopeComposite(model,source,[(WINDOW,IDEN)]),
              {"heads_masked":[],"blocks_masked":list(WINDOW)})
    gate("Cref reproduces W3 prefix_5 at block 11",label,
         abs(rcref["ssdc"][11]-A["prefix_5_b11"])<ANCHOR_TOL,
         f"{rcref['ssdc'][11]:+.6f} vs W3 {A['prefix_5_b11']}")
    for h in range(n_heads):
        run(f"C_head{h:02d}","C",
            RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h],"blocks_masked":list(WINDOW)+list(POST)})
    base_c=rcref["ssdc"][11]
    rankC=sorted(range(n_heads),
                 key=lambda h:(-(base_c-entry["conditions"][f"C_head{h:02d}"]["ssdc"][11]),h))
    entry["stageC_ranking"]=[{"head":h,
        "b11_drop_vs_Cref":base_c-entry["conditions"][f"C_head{h:02d}"]["ssdc"][11]} for h in rankC]
    print("  stage C ranking (biggest re-injection loss first):",rankC)

    # ---- STAGE D: are the regenerators axis-specific? ---------------------
    print("  stage D: axis window 0-5, plus a top regenerator off in 6-11")
    rdr=run("Dref_axisrow5","Dref",RopeComposite(model,source,[(WINDOW,ROWM)]),
            {"heads_masked":[],"blocks_masked":list(WINDOW),"axis":"row"})
    gate("Dref axisrow reproduces W3 at block 11",label,
         abs(rdr["ssdc"][11]-A["axisrow_5_b11"])<ANCHOR_TOL,
         f"{rdr['ssdc'][11]:+.6f} vs W3 {A['axisrow_5_b11']}")
    rdc=run("Dref_axiscol5","Dref",RopeComposite(model,source,[(WINDOW,COLM)]),
            {"heads_masked":[],"blocks_masked":list(WINDOW),"axis":"col"})
    gate("Dref axiscol reproduces W3 at block 11",label,
         abs(rdc["ssdc"][11]-A["axiscol_5_b11"])<ANCHOR_TOL,
         f"{rdc['ssdc'][11]:+.6f} vs W3 {A['axiscol_5_b11']}")
    for h in rankC[:D_TOP_N]:
        run(f"D_row_head{h:02d}","D",
            RopeComposite(model,source,[(WINDOW,ROWM),(POST,per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h],"blocks_masked":list(WINDOW)+list(POST),"axis":"row"})
        run(f"D_col_head{h:02d}","D",
            RopeComposite(model,source,[(WINDOW,COLM),(POST,per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h],"blocks_masked":list(WINDOW)+list(POST),"axis":"col"})

    # ---- hooks gate ------------------------------------------------------
    inert=[n for n,r in entry["conditions"].items()
           if n!="baseline" and r.get("hooks_registered",0)==0]
    gate("every intervening condition registered at least one hook",label,not inert,
         f"inert: {inert}" if inert else "")

    entry["runtime_s"]=round(time.time()-t_model,1)
    RESULTS[label]=entry
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()
    print(f"  model done in {entry['runtime_s']}s")

# ---- cell 21 ----
import pandas as pd
pd.set_option("display.width",240); pd.set_option("display.float_format",lambda v:f"{v:,.4f}")

print("STAGE A: which heads carry position (all blocks masked for one head)\n")
for lab,e in RESULTS.items():
    PL=e["probe_layer"]
    rows=[{"head":d["head"],"ssdc_ret@probe":d["ssdc_retention_at_probe"],
           "exact_probe":e["conditions"][f"A_head{d['head']:02d}"]["probes"].get(str(PL),{}).get("exact")}
          for d in e["stageA_ranking"]]
    print(f"--- {lab}"); print(pd.DataFrame(rows).to_string(index=False)); print()

print("\nSTAGE B: concentrated or spread? top-k against a random-k control\n")
for lab,e in RESULTS.items():
    PL=e["probe_layer"]; rows=[]
    for k in BK:
        r=e["conditions"][f"B_top{k}"]
        rows.append({"k":k,"heads":"top","ssdc_ret@probe":sret(e,r["ssdc"],PL),
                     "heads_masked":str(r["heads_masked"])})
    for k in BCTL_K:
        r=e["conditions"][f"Bctl_rand{k}"]
        rows.append({"k":k,"heads":"random","ssdc_ret@probe":sret(e,r["ssdc"],PL),
                     "heads_masked":str(r["heads_masked"])})
    print(f"--- {lab}")
    print(pd.DataFrame(rows).sort_values(["k","heads"]).to_string(index=False))
    print(f"    random draws overlap with top-k: "
          f"{ {k:v['overlap_with_topk'] for k,v in e['random_head_draws'].items()} }")
    print()

# ---- cell 22 ----
print("STAGE C: which heads perform the re-injection\n")
print("Cref masks rope in blocks 0-5. Each C condition additionally masks one head in 6-11.")
print("A large drop against Cref means that head was doing the rebuilding.\n")
for lab,e in RESULTS.items():
    cref=e["conditions"]["Cref_prefix5"]
    rows=[]
    for d in e["stageC_ranking"]:
        h=d["head"]; r=e["conditions"][f"C_head{h:02d}"]
        rows.append({"head":h,
                     "b11_ssdc":r["ssdc"][11],
                     "b11_ret":sret(e,r["ssdc"],11),
                     "drop_vs_Cref":d["b11_drop_vs_Cref"],
                     "pct_of_Cref_recovery":100*d["b11_drop_vs_Cref"]/max(cref["ssdc"][11]-e["floor_ssdc"][11],1e-9)})
    print(f"--- {lab}   Cref b11 = {cref['ssdc'][11]:.4f} (retention {sret(e,cref['ssdc'],11):.3f})")
    print(pd.DataFrame(rows).to_string(index=False)); print()

# ---- cell 23 ----
print("STAGE D: are the regenerators axis-specific?\n")
def pret(e,r,b,h):
    pb=e["conditions"]["baseline"]["probes"].get(str(b)); p=r.get("probes",{}).get(str(b))
    if not pb or not p: return float("nan")
    ch=p["chance"][h]; rng=pb[h]-ch
    return (p[h]-ch)/rng if rng>1e-6 else float("nan")

for lab,e in RESULTS.items():
    print(f"--- {lab}")
    for axis,ref in (("row","Dref_axisrow5"),("col","Dref_axiscol5")):
        rr=e["conditions"][ref]
        print(f"  {axis}-axis masked in 0-5.  reference {ref}: "
              f"b11 {axis}_ret={pret(e,rr,11,axis):.3f}")
        for cond,r in e["conditions"].items():
            if not cond.startswith(f"D_{axis}_head"): continue
            h=r["heads_masked"][0]
            print(f"    + head {h:2d} off in 6-11:  {axis}_ret={pret(e,r,11,axis):.3f}"
                  f"   (other axis {pret(e,r,11,'col' if axis=='row' else 'row'):.3f})"
                  f"   ssdc_ret={sret(e,r['ssdc'],11):.3f}")
        print()
print("Reading: if knocking out a head collapses the masked axis's recovery but not the")
print("other axis, that head is an axis-specific regenerator.")

# ---- cell 24 ----
ok=[(l,e) for l,e in RESULTS.items() if e.get("status")=="ok"]
if ok:
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(ok),figsize=(5.4*len(ok),4.0),sharey=True)
    axes=np.atleast_1d(axes)
    for ax,(lab,e) in zip(axes,ok):
        cref=e["conditions"]["Cref_prefix5"]
        bs=list(range(POST[0]+1,N_BLOCKS))
        ax.plot(bs,[sret(e,cref["ssdc"],b) for b in bs],"o-",color="black",lw=2.2,
                label="Cref (no head knocked out)")
        cm=plt.get_cmap("coolwarm")
        for i,d in enumerate(e["stageC_ranking"]):
            h=d["head"]; r=e["conditions"][f"C_head{h:02d}"]
            ax.plot(bs,[sret(e,r["ssdc"],b) for b in bs],"-",lw=1.2,alpha=0.85,
                    color=cm(i/max(len(e["stageC_ranking"])-1,1)),label=f"h{h}" if i<3 else None)
        ax.set_title(lab,fontsize=9); ax.set_xlabel("block"); ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("SSDC retention after the window")
    fig.suptitle("Stage C: knocking one head out of the re-injection segment",y=1.02)
    fig.tight_layout(); plt.show()

# ---- cell 26 ----
gdf=pd.DataFrame(GATES)
print("GATE REPORT\n")
print(gdf.to_string(index=False) if len(gdf) else "(none)")
n_fail=int((gdf.status=="FAIL").sum()) if len(gdf) else 0

# Gate accounting: a skipped gate is reported, not silently dropped.
# pre-model: 1 metric unit test + 8 per layout x 2 layouts + 2 composite = 19
# per-model: grid, rope-rows, num_heads, plane-decisive, axis-balanced, baseline anchor,
#            floor anchor, block-0, all-heads-vs-floor, stageA permutation, Cref repro,
#            Dref row, Dref col, hooks = 14
per_model_gates=14
expected=19+per_model_gates*len(MODELS)
print(f"\ngates run: {len(GATES)}   expected at least: {expected}")
acct = len(GATES)>=expected
print(f"gate accounting: {'OK' if acct else 'SHORT: some gates did not run'}")
print("\n"+"="*78)
print("ALL GATES PASSED" if n_fail==0 else f"{n_fail} GATE(S) FAILED: report before using")
if n_fail:
    for _,r in gdf[gdf.status=="FAIL"].iterrows():
        print(f"   - [{r['scope']}] {r['gate']}: {r['detail']}")
print("="*78)

# ---- cell 28 ----
export={
 "notebook":"W4","generated_utc":RUN_STAMP,
 "environment":{"torch":torch.__version__,"timm":timm.__version__,"device":DEVICE,
                "dtype":"float32","repo_path":str(repo),
                "gpu":torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"},
 "measurement":{"implementation":"repo metrics/ssdc.py evaluate_ssdc","RPI":True,
                "number_images":NUMBER_IMAGES,"batch_size":BATCH_SIZE,"metric":METRIC,
                "num_workers":0,"torch_seed":TORCH_SEED,"head_seed":HEAD_SEED,
                "input_size":INPUT_SIZE,
                "precision":"fp32; baseline and floor re-measured in-run"},
 "probe_settings":{"lr":PROBE_LR,"passes":PROBE_PASSES,"batch":PROBE_BATCH,
                   "train_frac":PROBE_TRAIN_FRAC,"split":"by image","standardised":True,
                   "heads":["exact","row","col"],"reported":"peak test accuracy over passes",
                   "device":PDEV,"depth_by_stage":PROBE_DEPTH},
 "comparison_policy":"intra-model only",
 "selection_rules":{"stage_B_from_A":RANK_A,"stage_D_from_C":RANK_C,
                    "fixed_before_run":True},
 "design_notes":{"window":WINDOW,"post":POST,
                 "readout_offset":"first recoverable block after a window ending at m is m+2",
                 "ape_excluded":"no per-head positional mechanism; DINOv3 also has num_classes=0 "
                                "so accuracy is unavailable for it"},
 "w3_anchors":W3_ANCHORS,
 "results":RESULTS,"gates":GATES,
 "gate_summary":{"total":len(GATES),"pass":int(sum(g["status"]=="PASS" for g in GATES)),
                 "fail":n_fail,"expected_at_least":expected,"accounting_ok":bool(acct)},
 "normalisation_protocol":{"formula":"(observed_b - floor_b) / (baseline_b - floor_b), per block",
                           "source":"in-run fp32 baseline and floor",
                           "floors":{l:e.get("floor_ssdc") for l,e in RESULTS.items()}},
}
out_json=OUT_DIR/f"w4_heads_{RUN_STAMP.replace(':','-')}.json"
out_json.write_text(json.dumps(export,indent=2))

rows=[]
for lab,e in RESULTS.items():
    for nm,r in e["conditions"].items():
        rows.append({"run":lab,"condition":nm,"kind":r["kind"],
                     "heads_masked":str(r.get("heads_masked","")),
                     "ssdc_probe":r["ssdc"][e["probe_layer"]],"ssdc_b11":r["ssdc"][11],
                     "ret_probe":sret(e,r["ssdc"],e["probe_layer"]),"ret_b11":sret(e,r["ssdc"],11),
                     "hooks":r.get("hooks_registered"),"runtime_s":r.get("runtime_s")})
pd.DataFrame(rows).to_csv(OUT_DIR/"w4_summary.csv",index=False)
pd.DataFrame([{"run":l,**d} for l,e in RESULTS.items() for d in e["stageA_ranking"]]
             ).to_csv(OUT_DIR/"w4_stageA_ranking.csv",index=False)
pd.DataFrame([{"run":l,**d} for l,e in RESULTS.items() for d in e["stageC_ranking"]]
             ).to_csv(OUT_DIR/"w4_stageC_ranking.csv",index=False)
gdf.to_csv(OUT_DIR/"w4_gate_report.csv",index=False)
pb=[{"run":l,"condition":n,"block":b,"ssdc":v}
    for l,e in RESULTS.items() for n,r in e["conditions"].items() for b,v in enumerate(r["ssdc"])]
pd.DataFrame(pb).to_csv(OUT_DIR/"w4_per_block.csv",index=False)

import shutil
archive=shutil.make_archive("w4_output","zip",OUT_DIR)
print("written:")
for p in sorted(OUT_DIR.iterdir()): print(f"  {p}  ({p.stat().st_size:,} bytes)")
print(f"\narchive: {archive} ({os.path.getsize(archive):,} bytes)")
try:
    from google.colab import files as cf; cf.download(archive)
except Exception:
    print("(not Colab: copy the archive back manually)")