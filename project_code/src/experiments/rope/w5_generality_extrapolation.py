# ---- cell 2 ----
import importlib, subprocess, sys
def ensure(p,m=None):
    try: importlib.import_module(m or p); return False
    except ImportError:
        print(f"installing {p} ..."); subprocess.check_call([sys.executable,"-m","pip","install","-q",p]); return True
for p in ["timm","transformers","datasets","scipy"]: ensure(p)
print("environment ready")

# ---- cell 3 ----
import os
import os, json, math, time, warnings, itertools
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
warnings.filterwarnings("ignore", category=UserWarning)
print("torch",torch.__version__,"| timm",timm.__version__,"| cuda",torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:",torch.cuda.get_device_name(0))
    free,total=torch.cuda.mem_get_info(); print(f"memory: {free/1e9:.1f} / {total/1e9:.1f} GB free")

# ---- cell 5 ----
# ============================================================================
# CONFIG
# ============================================================================
REPO_URL  = None
REPO_PATH = str(Path(__file__).resolve().parents[2])

PARTS = ["A","B","C","D","E"]        # drop letters to run a subset

NUMBER_IMAGES = 1000
BATCH_SIZE    = 256
METRIC        = "manhattan"
TORCH_SEED    = 20260908
HEAD_SEED     = 4127
BASE_RES      = 224
EXTRAP_RES    = 320                  # 20x20 = 400 tokens
ATTN_IMAGES   = 64                   # part E only; full attention maps are large
ATTN_BATCH    = 32

PROBE_LR, PROBE_PASSES, PROBE_BATCH, PROBE_TRAIN_FRAC = 5e-3, 20, 512, 0.8
PROBE_CHUNK  = 8192
PROBE_DEVICE = None

# --- part B: probe error bars, computed on cached activations (no extra forward passes)
NOISE_SEEDS   = [0, 1, 2]
NOISE_BOOT    = 200
NOISE_CONDS   = ["baseline", "Dref_axisrow5", "Dref_axiscol5"]
NOISE_BLOCKS  = [7, 11]              # block-to-block is the bigger term, so use >1
NOISE_MODELS  = ["RoPE · naver", "RoPE · DINOv3"]   # where the stage-D claim lives

ANCHOR_TOL = 0.01
REPRO_TOL  = 1e-6                    # tolerance for reproducing the SSDC anchors
EQUIV_TOL  = 1e-9

WINDOW = list(range(0,6)); POST = list(range(6,12))
D_TOP_12HEAD, D_TOP_6HEAD = 4, 3     # stage C heads carried into stage D

# stage D readout: mean over the post-window blocks
D_READOUT = "mean_over_post"

RANK_A = "lowest SSDC retention at the probe layer (all-blocks single-head mask); ties by head index"
RANK_C = "largest drop in block-11 SSDC retention vs the prefix_5 anchor; ties by head index"

# probe blocks per condition; baseline covers every block used downstream as a denominator
PROBE_DEPTH = {"baseline":"union","floor":"probe_layer","axis":"sparse","win":"post",
               "A":"probe_layer","C":"none","D":"post","res":"pair","acc":"none"}

# --- models -----------------------------------------------------------------
# rope_mixed is deliberately excluded: its learned frequencies are (2,12,12,32) and most
# planes are oblique (median |smaller|/|larger| = 0.225, 28% above 0.5), so the row/col
# plane partition that W2 and stage D rest on does not exist for it.
MODELS = {
 "RoPE · naver":   dict(kwargs=dict(),                                                    probe_layer=4, hybrid=False),
 "RoPE · DINOv3":  dict(kwargs=dict(model_name="vit_base_patch16_dinov3.lvd1689m"),        probe_layer=4, hybrid=False),
 "RoPE · small":   dict(kwargs=dict(model_name="vit_small_patch16_rope_224.naver_in1k"),   probe_layer=4, hybrid=False),
 "RoPE · rope_ape":dict(kwargs=dict(model_name="vit_base_patch16_rope_ape_224.naver_in1k"),probe_layer=4, hybrid=True),
 "RoPE · reg1_gap":dict(kwargs=dict(model_name="vit_base_patch16_rope_reg1_gap_256.sbb_in1k"),probe_layer=4, hybrid=False),
}
# reg1_gap_256 is trained at 256 and run at 224 like the others, through load_rope's
# dynamic_img_size.

RES_MODELS = ["RoPE · naver","RoPE · rope_ape","RoPE · reg1_gap","RoPE · small"]  # have classifiers
ACC_MODELS = RES_MODELS

# SSDC anchors from W3/W4 (fp32, same protocol)
ANCHORS = {
 "RoPE · naver":  dict(baseline=0.469422, floor=0.012865, prefix_5_b11=0.329663,
                       axisrow_5_b11=0.382360, axiscol_5_b11=0.419516, n_heads=12),
 "RoPE · DINOv3": dict(baseline=0.948654, floor=0.017712, prefix_5_b11=0.442640,
                       axisrow_5_b11=0.627570, axiscol_5_b11=0.652320, n_heads=12),
}

OUT_DIR=Path("w5_output"); OUT_DIR.mkdir(exist_ok=True)
# ============================================================================

RUN_STAMP=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
PDEV=PROBE_DEVICE or DEVICE
N_BLOCKS=12
GATES=[]
def gate(name,scope,passed,detail=""):
    GATES.append({"gate":name,"scope":scope,"status":"PASS" if passed else "FAIL","detail":detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {scope}: {name}"+(f": {detail}" if detail else ""))
    return passed
print(f"device {DEVICE} | probes {PDEV} | fp32 | parts {PARTS}")
print(f"resolutions {BASE_RES} and {EXTRAP_RES} | stage-D readout: {D_READOUT}")
print(f"run {RUN_STAMP}")

# ---- cell 7 ----
def resolve_repo():
    c=[]
    if REPO_PATH: c.append(Path(REPO_PATH).expanduser().resolve())
    cl=Path("vit-sae-analysis"); c+= [cl/"project_code"/"src", cl/"src", cl]
    for x in c:
        if (x/"metrics").exists() and (x/"main").exists(): return x
    if REPO_URL and not cl.exists():
        subprocess.check_call(["git","clone","--depth","1",REPO_URL,str(cl)])
        for x in c:
            if (x/"metrics").exists() and (x/"main").exists(): return x
    raise FileNotFoundError("no checkout with metrics/ and main/: "+", ".join(map(str,c)))
repo=resolve_repo()
if str(repo) not in sys.path: sys.path.insert(0,str(repo))
print("repo:",repo)

from metrics.ssdc import evaluate_ssdc, spatial_similarity_distance_correlation
from main.load_models import (get_vit_blocks, get_block_attention, get_patch_embed_conv, load_rope)
from main.model import predict
from main.prep_data import prep_data
from experiments.common import load_imagenet

from scipy.spatial.distance import cdist as _cdist
_G=14; _c=np.stack(np.meshgrid(np.arange(_G),np.arange(_G),indexing="ij"),-1).reshape(-1,2)
_D=_cdist(_c,_c,metric="cityblock")
_a=spatial_similarity_distance_correlation(-_D,grid_size=_G,metric="manhattan")
_b=spatial_similarity_distance_correlation(_D,grid_size=_G,metric="manhattan")
_r=spatial_similarity_distance_correlation(np.random.default_rng(0).normal(size=(_G*_G,)*2),
                                           grid_size=_G,metric="manhattan")
gate("imported metric satisfies its unit test","repo-metric",
     _a>0.999 and _b<-0.999 and abs(_r)<0.1,f"{_a:+.3f}/{_b:+.3f}/{_r:+.3f}")
DATASET=load_imagenet(split="validation",streaming=True)
print("dataset ready:",type(DATASET).__name__)

# ---- cell 9 ----
def find_rope(model):
    for n,m in model.named_modules():
        if callable(getattr(m,"get_embed",None)): return n,m
    return None,None

def forward_grid(model,processor,source,res):
    g={}
    conv=get_patch_embed_conv(model,source)
    h=conv.register_forward_hook(lambda m,i,o: g.update(hw=(int(o.shape[2]),int(o.shape[3])) if o.dim()==4 else None))
    dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=BATCH_SIZE,
                 batch_size=BATCH_SIZE,half=False,num_workers=0)
    with torch.no_grad():
        for im,_ in dl: model(im.to(DEVICE)); break
    h.remove(); return g.get("hw")

def attn_attrs(model,source):
    rh,src,nh,npt=False,"default(interleaved)",None,None
    for blk in get_vit_blocks(model,source):
        a=get_block_attention(blk,source)
        for at in ("rotate_half","half","rope_rotate_half"):
            v=getattr(a,at,None)
            if isinstance(v,bool): rh,src=v,at; break
        nh=getattr(a,"num_heads",None); npt=getattr(a,"num_prefix_tokens",None); break
    return rh,src,nh,npt

def plane_columns(p,n,rh): return [p,p+n] if rh else [2*p,2*p+1]

def derive_axis(emb,n_planes,rh,H,W):
    sin,cos=emb.detach().float().chunk(2,-1); rows=[]; dec=True
    for p in range(n_planes):
        c=plane_columns(p,n_planes,rh)[0]
        z=torch.complex(cos[:,c],sin[:,c]).reshape(H,W)
        ra=(z-z.mean(dim=1,keepdim=True)).abs().mean().item()
        rb=(z-z.mean(dim=0,keepdim=True)).abs().mean().item()
        ax="row" if ra<rb else "col"; lo,hi=sorted([ra,rb]); r=lo/hi if hi>1e-12 else 1.0
        if r>0.10: dec=False
        rows.append({"plane":p,"axis":ax,"ratio":r})
    return rows,[r["plane"] for r in rows if r["axis"]=="row"],[r["plane"] for r in rows if r["axis"]=="col"],dec

def mask_planes(emb,kill,n,rh):
    if not kill: return emb
    s,c=emb.chunk(2,-1); s,c=s.clone(),c.clone()
    for p in kill:
        for col in plane_columns(p,n,rh): s[...,col]=0.0; c[...,col]=1.0
    return torch.cat([s,c],-1)

def identity_like(emb):
    s,c=emb.chunk(2,-1); return torch.cat([torch.zeros_like(s),torch.ones_like(c)],-1)

def per_head(heads,inner,n_heads):
    hs=sorted(set(heads))
    def tf(emb):
        if emb.dim()!=2:
            raise RuntimeError(f"per_head expected 2-D rope, got {tuple(emb.shape)}: stacked overrides")
        out=emb.unsqueeze(0).repeat(n_heads,1,1)
        if hs:
            t=inner(emb)
            for h in hs: out[h]=t
        return out
    return tf

class RopeGlobal:
    def __init__(s,model,tf): s.name,s.mod=find_rope(model); s.tf=tf; s.handles=[]
    def __enter__(s):
        if s.mod is None: raise RuntimeError("no get_embed module")
        s.orig=s.mod.get_embed; o,t=s.orig,s.tf
        s.mod.get_embed=(lambda _o=o,_t=t:(lambda *a,**k:_t(_o(*a,**k))))()
        s.handles=[1]; return s
    def __exit__(s,*e): s.mod.get_embed=s.orig; s.handles=[]

class RopeComposite:
    def __init__(s,model,source,segs):
        s.blocks=get_vit_blocks(model,source); s.source=source
        s.segs=[(list(b),t) for b,t in segs]; seen=set()
        for b,_ in s.segs:
            if seen & set(b): raise ValueError(f"overlapping segments: {sorted(seen&set(b))}")
            seen|=set(b)
        s.handles=[]
    def __enter__(s):
        def mk(tf):
            def pre(mod,args,kw):
                r=kw.get("rope",None)
                if r is not None: kw=dict(kw); kw["rope"]=tf(r)
                return args,kw
            return pre
        for blocks,tf in s.segs:
            for i in blocks:
                s.handles.append(get_block_attention(s.blocks[i],s.source)
                                 .register_forward_pre_hook(mk(tf),with_kwargs=True))
        return s
    def __exit__(s,*e):
        for h in s.handles: h.remove()
        s.handles=[]

class APEZero:
    def __init__(s,model): s.model=model; s.handles=[]
    def __enter__(s):
        cands=[n for n,_ in s.model.named_parameters()
               if n.endswith("pos_embed") or n.endswith("position_embeddings")]
        if not cands: raise RuntimeError("no positional-embedding parameter")
        s.name=sorted(cands,key=len)[0]; parts=s.name.split(".")
        m=s.model
        for p in parts[:-1]: m=getattr(m,p)
        s.parent,s.attr=m,parts[-1]
        pe=getattr(s.parent,s.attr); s.saved=pe.detach().clone()
        with torch.no_grad(): pe.zero_()
        s.handles=[1]; return s
    def __exit__(s,*e):
        with torch.no_grad(): getattr(s.parent,s.attr).copy_(s.saved)
        s.handles=[]

class Both:
    '''rope identity AND pos_embed zeroed: the true no-position floor for a hybrid.'''
    def __init__(s,model,tf): s.a=RopeGlobal(model,tf); s.b=APEZero(model); s.handles=[]
    def __enter__(s): s.a.__enter__(); s.b.__enter__(); s.handles=[1,1]; return s
    def __exit__(s,*e): s.b.__exit__(); s.a.__exit__(); s.handles=[]

class NullCtx:
    handles=[]
    def __enter__(s): return s
    def __exit__(s,*e): return False

# ---- cell 11 ----
def _rot_il(x):
    x=x.unflatten(-1,(-1,2)); return torch.stack([-x[...,1],x[...,0]],-1).flatten(-2)
def _rot_h(x):
    a,b=x.chunk(2,-1); return torch.cat([-b,a],-1)
def _apply(x,e,half=False):
    s,c=e.chunk(2,-1); return x*c+(_rot_h(x) if half else _rot_il(x))*s

for lname,rh in [("interleaved",False),("rotate_half",True)]:
    npl,dim,H=32,64,12
    e=torch.randn(196,2*dim); x=torch.randn(2,H,196,dim)
    gate(f"empty plane mask bit-identical [{lname}]","algebra",torch.equal(mask_planes(e,[],npl,rh),e))
    gate(f"full plane mask equals identity [{lname}]","algebra",
         torch.allclose(mask_planes(e,list(range(npl)),npl,rh),identity_like(e)))
    m=mask_planes(e,[3],npl,rh)
    touched=torch.nonzero((m-e).abs().sum(0)>0).flatten().tolist()
    sc=plane_columns(3,npl,rh); exp=sorted(set(sc+[c+dim for c in sc]))
    gate(f"plane 3 touches only its own columns [{lname}]","algebra",touched==exp,f"{touched} vs {exp}")
    full=_apply(x,e,rh); iden=_apply(x,identity_like(e),rh)
    gate(f"mask NO heads equals unmodified rope [{lname}]","per-head",
         torch.allclose(_apply(x,per_head([],identity_like,H)(e),rh),full,atol=1e-6))
    gate(f"mask ALL heads equals global identity [{lname}]","per-head",
         torch.allclose(_apply(x,per_head(range(H),identity_like,H)(e),rh),iden,atol=1e-6))
    one=_apply(x,per_head([3],identity_like,H)(e),rh); others=[h for h in range(H) if h!=3]
    gate(f"mask head 3 leaves others untouched [{lname}]","per-head",
         torch.allclose(one[:,others],full[:,others],atol=1e-6))
    gate(f"mask head 3 changes head 3 [{lname}]","per-head",not torch.allclose(one[:,3],full[:,3]))

class _Stub:
    def __init__(s,n=12): s.blocks=[nn.Identity() for _ in range(n)]
try:
    RopeComposite(_Stub(),"timm",[(range(0,6),identity_like),(range(5,12),identity_like)]); ov=False
except ValueError as ex: ov="overlapping" in str(ex)
gate("composite REJECTS overlapping segments","composite",ov)
try:
    RopeComposite(_Stub(),"timm",[(range(0,6),identity_like),(range(6,12),identity_like)]); dj=True
except Exception: dj=False
gate("composite ACCEPTS disjoint segments","composite",dj)

# ---- cell 13 ----
class Capture:
    def __init__(s,model,source,blocks,n_prefix):
        s.blocks=get_vit_blocks(model,source); s.which=list(blocks)
        s.n_prefix=int(n_prefix); s.source=source; s.store={i:[] for i in s.which}; s.handles=[]
    def __enter__(s):
        def mk(i):
            def hook(mod,inp,out):
                t=inp[0].detach()
                if s.n_prefix>0: t=t[:,s.n_prefix:,:]
                s.store[i].append(t.float().cpu())
            return hook
        for i in s.which:
            s.handles.append(get_block_attention(s.blocks[i],s.source).register_forward_hook(mk(i)))
        return s
    def __exit__(s,*e):
        for h in s.handles: h.remove()
        s.handles=[]
    def stacked(s,i): return torch.cat(s.store[i],0) if s.store[i] else None

def _fit(Xtr,ytr,Xte,yte,nc,seed):
    torch.manual_seed(seed)
    head=nn.Linear(Xtr.shape[1],nc).to(Xtr.device)
    opt=torch.optim.AdamW(head.parameters(),lr=PROBE_LR)
    n=Xtr.shape[0]; peak=0.0; best=None
    for _ in range(PROBE_PASSES):
        perm=torch.randperm(n,device=Xtr.device); head.train()
        for s0 in range(0,n,PROBE_BATCH):
            b=perm[s0:s0+PROBE_BATCH]
            opt.zero_grad(); F.cross_entropy(head(Xtr[b]),ytr[b]).backward(); opt.step()
        head.eval(); correct=0; preds=[]
        with torch.no_grad():
            for s0 in range(0,Xte.shape[0],PROBE_CHUNK):
                p=head(Xte[s0:s0+PROBE_CHUNK]).argmax(-1); preds.append(p)
                correct+=(p==yte[s0:s0+PROBE_CHUNK]).sum().item()
        acc=correct/Xte.shape[0]
        if acc>peak: peak=acc; best=torch.cat(preds)
    del head,opt
    return float(peak), best

def _split(n_images,T,seed=TORCH_SEED):
    g=torch.Generator().manual_seed(seed)
    idx=torch.randperm(n_images,generator=g); n_tr=int(round(n_images*PROBE_TRAIN_FRAC))
    rows=lambda ids:(ids.unsqueeze(1)*T+torch.arange(T).unsqueeze(0)).reshape(-1)
    return rows(idx[:n_tr]),rows(idx[n_tr:]),idx[n_tr:]

def probe_block(feats,H,W,n_images,block,noise=False):
    '''Three heads from one transfer. Seeded per block.'''
    T=H*W; tr,te,te_img=_split(n_images,T)
    Xtr=feats[tr].to(PDEV); Xte=feats[te].to(PDEV)
    mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True).clamp_min(1e-6)
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    slot=torch.arange(T); out={}
    for name,lab,nc in (("exact",slot,T),("row",slot//W,H),("col",slot%W,W)):
        y=lab.repeat(n_images)
        acc,preds=_fit(Xtr,y[tr].to(PDEV),Xte,y[te].to(PDEV),nc,TORCH_SEED+block)
        out[name]=acc
        if noise:
            corr=(preds==y[te].to(PDEV)).float()
            g=np.random.default_rng(TORCH_SEED+block)
            n_te=len(te_img); per=corr.reshape(n_te,T).mean(1).cpu().numpy()
            boot=[per[g.integers(0,n_te,n_te)].mean() for _ in range(NOISE_BOOT)]
            seeds=[_fit(Xtr,y[tr].to(PDEV),Xte,y[te].to(PDEV),nc,TORCH_SEED+block+1000*s)[0]
                   for s in NOISE_SEEDS]
            out.setdefault("noise",{})[name]={
                "boot_sd":float(np.std(boot,ddof=1)),"boot_mean":float(np.mean(boot)),
                "seed_sd":float(np.std(seeds,ddof=1)),"seed_values":[float(v) for v in seeds],
                "combined_sd":float(np.sqrt(np.var(boot,ddof=1)+np.var(seeds,ddof=1)))}
    out["chance"]={"exact":1.0/T,"row":1.0/H,"col":1.0/W}
    del Xtr,Xte
    if PDEV=="cuda": torch.cuda.empty_cache()
    return out

# ---- cell 15 ----
def depth_blocks(kind,PL):
    d=PROBE_DEPTH[kind]
    if d=="all": return list(range(N_BLOCKS))
    if d=="none": return []
    if d=="probe_layer": return [PL]
    if d=="last": return [N_BLOCKS-1]
    if d=="sparse": return sorted({PL,min(PL+2,N_BLOCKS-1),min(PL+4,N_BLOCKS-1),N_BLOCKS-1})
    if d=="pair": return sorted({PL,N_BLOCKS-1})
    if d=="union":
        u={PL,min(PL+2,N_BLOCKS-1),min(PL+4,N_BLOCKS-1),N_BLOCKS-1}
        u|=set(range(POST[0]+1,N_BLOCKS))
        return sorted(u)
    if d=="post": return [b for b in range(POST[0]+1,N_BLOCKS)]
    raise ValueError(d)

def measure(model,processor,source,ctx,probe_blocks,n_prefix,H,W,noise=False):
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
            a=cap.stacked(i)
            if a is None: continue
            want=noise and (i in NOISE_BLOCKS)
            res["probes"][str(i)]=probe_block(a.reshape(-1,a.shape[-1]),H,W,a.shape[0],i,noise=want)
            del a
        cap.store.clear()
    res["runtime_s"]=round(time.time()-t0,1)
    return res

def accuracy(model,processor,source,ctx):
    out={}
    for tag,rpi in (("clean",False),("rpi",True)):
        torch.manual_seed(TORCH_SEED)
        dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=NUMBER_IMAGES,
                     batch_size=BATCH_SIZE,half=False,num_workers=0)
        with ctx: out[tag]=float(predict(model,dl,source,RPI=rpi,half=False))
    return out

def sret(e,ssdc,b):
    rng=e["baseline_ssdc"][b]-e["floor_ssdc"][b]
    return (ssdc[b]-e["floor_ssdc"][b])/rng if abs(rng)>1e-6 else float("nan")

def save(tag,obj):
    p=OUT_DIR/f"w5_{tag}_{RUN_STAMP.replace(':','-')}.json"
    p.write_text(json.dumps(obj,indent=2)); print(f"    saved -> {p.name}")

# ---- cell 17 ----
RESULTS={}
if "A" in PARTS:
  for label,cfg in MODELS.items():
    print("\n"+"="*78); print(label); print("="*78)
    t0=time.time(); PL=cfg["probe_layer"]
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**cfg["kwargs"])
    model.eval().float()
    rh,rh_src,nh_attr,_=attn_attrs(model,source)
    n_prefix=int(getattr(model,"num_prefix_tokens",1))
    hw=forward_grid(model,processor,source,BASE_RES)
    Hg,Wg=hw; T=Hg*Wg
    n_heads=int(nh_attr) if nh_attr else 12
    rname,rmod=find_rope(model); emb0=rmod.get_embed(shape=(Hg,Wg))
    n_planes=int(emb0.shape[-1]//4)
    amap,rowp,colp,dec=derive_axis(emb0,n_planes,rh,Hg,Wg)
    D_TOP=D_TOP_12HEAD if n_heads>=12 else D_TOP_6HEAD
    print(f"  source={source} heads={n_heads} prefix={n_prefix} grid={Hg}x{Wg} planes={n_planes} "
          f"rotate_half={rh} hybrid={cfg['hybrid']}")
    gate("forward grid is 14x14",label,(Hg,Wg)==(14,14),f"{Hg}x{Wg}")
    gate("rope rows equal patch tokens",label,int(emb0.shape[0])==T,f"{emb0.shape[0]} vs {T}")
    gate("every plane classifies decisively",label,dec,f"max ratio {max(a['ratio'] for a in amap):.1e}")
    gate("axis split is balanced",label,len(rowp)==len(colp)==n_planes//2,f"{len(rowp)}/{len(colp)}")

    e={"status":"ok","source":source,"probe_layer":PL,"n_heads":n_heads,"n_planes":n_planes,
       "rotate_half":bool(rh),"rotate_half_source":rh_src,"num_prefix_tokens":n_prefix,
       "grid":[Hg,Wg],"patch_tokens":T,"hybrid":cfg["hybrid"],"row_planes":rowp,"col_planes":colp,
       "axis_map":amap,"d_top":D_TOP,"d_readout":D_READOUT,"resolution":BASE_RES,"conditions":{}}
    IDEN=identity_like
    ROWM=lambda x: mask_planes(x,rowp,n_planes,rh)
    COLM=lambda x: mask_planes(x,colp,n_planes,rh)

    def run(name,kind,ctx,extra=None,noise=False):
        pb=depth_blocks(kind,PL)
        r=measure(model,processor,source,ctx,pb,n_prefix,Hg,Wg,noise=noise)
        r["kind"]=kind; r["probe_blocks"]=pb
        if extra: r.update(extra)
        e["conditions"][name]=r
        print(f"    {name:16s} ssdc@{PL}={r['ssdc'][PL]:+.4f} b11={r['ssdc'][11]:+.4f} "
              f"hooks={r['hooks_registered']:2d} {r['runtime_s']}s")
        return r

    want_noise = ("B" in PARTS) and (label in NOISE_MODELS)
    rb=run("baseline","baseline",NullCtx(),noise=want_noise); e["baseline_ssdc"]=rb["ssdc"]
    rf=run("floor","floor",RopeGlobal(model,IDEN));            e["floor_ssdc"]=rf["ssdc"]
    if cfg["hybrid"]:
        run("ape_floor","floor",APEZero(model))
        run("both_floor","floor",Both(model,IDEN))

    A=ANCHORS.get(label)
    if A:
        gate("baseline hits its W3/W4 anchor",label,abs(rb["ssdc"][PL]-A["baseline"])<ANCHOR_TOL,
             f"{rb['ssdc'][PL]:+.6f} vs {A['baseline']}")
        gate("floor hits its W3/W4 anchor",label,abs(rf["ssdc"][PL]-A["floor"])<ANCHOR_TOL,
             f"{rf['ssdc'][PL]:+.6f} vs {A['floor']}")
    gate("floor is below baseline",label,rf["ssdc"][PL]<rb["ssdc"][PL]-0.01)
    gate("floor equals baseline at block 0",label,abs(rb["ssdc"][0]-rf["ssdc"][0])<1e-9,
         f"delta {abs(rb['ssdc'][0]-rf['ssdc'][0]):.2e}")

    rall=measure(model,processor,source,
                 RopeComposite(model,source,[(range(N_BLOCKS),per_head(range(n_heads),IDEN,n_heads))]),
                 [],n_prefix,Hg,Wg)
    dmax=max(abs(a-b) for a,b in zip(rall["ssdc"],rf["ssdc"]))
    gate("per-head mask of ALL heads reproduces the global floor",label,dmax<EQUIV_TOL,f"{dmax:.2e}")

    # --- W2 replication: global axis masking -------------------------------
    run("axis_row","axis",RopeGlobal(model,ROWM),{"axis":"row"})
    run("axis_col","axis",RopeGlobal(model,COLM),{"axis":"col"})

    # --- W3 replication: windowed ------------------------------------------
    rc=run("Cref_prefix5","win",RopeComposite(model,source,[(WINDOW,IDEN)]),{"heads_masked":[]})
    rdr=run("Dref_axisrow5","win",RopeComposite(model,source,[(WINDOW,ROWM)]),
            {"axis":"row","heads_masked":[]},noise=want_noise)
    rdc=run("Dref_axiscol5","win",RopeComposite(model,source,[(WINDOW,COLM)]),
            {"axis":"col","heads_masked":[]},noise=want_noise)
    if A:
        for nm,val,key in (("Cref_prefix5",rc,"prefix_5_b11"),("Dref_axisrow5",rdr,"axisrow_5_b11"),
                           ("Dref_axiscol5",rdc,"axiscol_5_b11")):
            gate(f"{nm} REPRODUCES W3 at block 11 (tol {REPRO_TOL})",label,
                 abs(val["ssdc"][11]-A[key])<REPRO_TOL,
                 f"{val['ssdc'][11]:.6f} vs {A[key]}  delta {abs(val['ssdc'][11]-A[key]):.2e}")

    # --- stage A ------------------------------------------------------------
    for h in range(n_heads):
        run(f"A_head{h:02d}","A",
            RopeComposite(model,source,[(range(N_BLOCKS),per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h]})
    rankA=sorted(range(n_heads),key=lambda h:(sret(e,e["conditions"][f"A_head{h:02d}"]["ssdc"],PL),h))
    e["stageA_ranking"]=[{"head":h,"ssdc_retention_at_probe":sret(e,e["conditions"][f"A_head{h:02d}"]["ssdc"],PL)} for h in rankA]
    print("  stage A ranking:",rankA)

    # --- stage C ------------------------------------------------------------
    for h in range(n_heads):
        run(f"C_head{h:02d}","C",
            RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h]})
    base_c=rc["ssdc"][11]
    rankC=sorted(range(n_heads),key=lambda h:(-(base_c-e["conditions"][f"C_head{h:02d}"]["ssdc"][11]),h))
    e["stageC_ranking"]=[{"head":h,"b11_drop_vs_Cref":base_c-e["conditions"][f"C_head{h:02d}"]["ssdc"][11]} for h in rankC]
    print("  stage C ranking:",rankC)

    # --- stage D, top-N, both axes -----------------------------------------
    for h in rankC[:D_TOP]:
        run(f"D_row_head{h:02d}","D",
            RopeComposite(model,source,[(WINDOW,ROWM),(POST,per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h],"axis":"row"})
        run(f"D_col_head{h:02d}","D",
            RopeComposite(model,source,[(WINDOW,COLM),(POST,per_head([h],IDEN,n_heads))]),
            {"heads_masked":[h],"axis":"col"})

    inert=[n for n,r in e["conditions"].items() if n!="baseline" and r.get("hooks_registered",0)==0]
    gate("every intervening condition registered a hook",label,not inert,f"inert: {inert}" if inert else "")
    e["runtime_s"]=round(time.time()-t0,1)
    RESULTS[label]=e
    save(f"partA_{label.split('·')[1].strip()}",e)          # incremental, per model
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()
    print(f"  model done in {e['runtime_s']}s")

# ---- cell 19 ----
RES={}
if "C" in PARTS:
  for label in RES_MODELS:
    if label not in RESULTS:
        print(f"  skipping {label}: part A result needed for the regenerator identity"); continue
    cfg=MODELS[label]; PL=cfg["probe_layer"]
    topC=RESULTS[label]["stageC_ranking"][0]["head"]
    nh=RESULTS[label]["n_heads"]
    rng_h=np.random.default_rng(HEAD_SEED)
    randh=int(rng_h.choice([h for h in range(nh) if h!=topC]))
    print("\n"+"="*78); print(f"{label}: resolution arm (regen head {topC}, random head {randh})")
    print("="*78)
    RES[label]={"top_regenerator":topC,"random_head":randh,"by_res":{}}
    for res in (BASE_RES,EXTRAP_RES):
        model,processor,source=load_rope(device=DEVICE,half=False,input_size=res,**cfg["kwargs"])
        model.eval().float()
        rh,_,nh_a,_=attn_attrs(model,source); n_heads=int(nh_a) if nh_a else nh
        n_prefix=int(getattr(model,"num_prefix_tokens",1))
        Hg,Wg=forward_grid(model,processor,source,res); T=Hg*Wg
        print(f"  res {res}: grid {Hg}x{Wg} = {T} tokens")
        gate(f"grid matches resolution {res}",label,Hg==res//16 and Wg==res//16,f"{Hg}x{Wg}")
        IDEN=identity_like
        block={}
        def rrun(name,ctx,want_acc=True):
            r=measure(model,processor,source,ctx,depth_blocks("res",PL),n_prefix,Hg,Wg)
            if want_acc and "D" in PARTS:
                try: r["acc"]=accuracy(model,processor,source,ctx)
                except Exception as ex: r["acc"]={"error":str(ex)}
            block[name]=r
            a=r.get("acc",{})
            print(f"    {name:14s} ssdc@{PL}={r['ssdc'][PL]:+.4f} b11={r['ssdc'][11]:+.4f}"
                  + (f"  clean={a.get('clean',float('nan')):.4f} rpi={a.get('rpi',float('nan')):.4f}"
                     if isinstance(a,dict) and "clean" in a else ""))
            return r
        rb=rrun("baseline",NullCtx()); rf=rrun("floor",RopeGlobal(model,IDEN))
        rrun("regen_ko",RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([topC],IDEN,n_heads))]))
        rrun("random_ko",RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([randh],IDEN,n_heads))]))
        rrun("Cref",RopeComposite(model,source,[(WINDOW,IDEN)]))
        gate(f"floor below baseline at {res}",label,rf["ssdc"][PL]<rb["ssdc"][PL]-0.01)
        RES[label]["by_res"][str(res)]={"grid":[Hg,Wg],"tokens":T,"conditions":block}
        del model
        if DEVICE=="cuda": torch.cuda.empty_cache()
    save(f"partC_{label.split('·')[1].strip()}",RES[label])

# ---- cell 21 ----
# attn_qkv rebuilds q, k, v the way AttentionRope / EvaAttention do, including rope and the
# separate q/k/v biases (reg1_gap). A gate checks the rebuild against the module's own output.
from timm.layers import apply_rot_embed_cat

def attn_qkv(mod,x,rope,npt):
    B,N,C=x.shape; nh=mod.num_heads; hd=C//nh
    if getattr(mod,"q_bias",None) is None:
        qkv=mod.qkv(x)
    else:
        bias=torch.cat((mod.q_bias,mod.k_bias,mod.v_bias))
        qkv=(mod.qkv(x)+bias if getattr(mod,"qkv_bias_separate",False)
             else F.linear(x,mod.qkv.weight,bias))
    q,k,v=qkv.reshape(B,N,3,nh,hd).permute(2,0,3,1,4).unbind(0)
    q,k=mod.q_norm(q),mod.k_norm(k)
    if rope is not None:
        half=getattr(mod,"rotate_half",False)
        q=torch.cat([q[:,:,:npt],apply_rot_embed_cat(q[:,:,npt:],rope,half=half)],2).type_as(v)
        k=torch.cat([k[:,:,:npt],apply_rot_embed_cat(k[:,:,npt:],rope,half=half)],2).type_as(v)
    return q,k,v

def attn_maps(mod,x,rope,npt):
    q,k,v=attn_qkv(mod,x,rope,npt)
    return ((q*mod.scale)@k.transpose(-2,-1)).softmax(-1)

ATTN={}
if "E" in PARTS:
  for label,cfg in MODELS.items():
    if label not in RESULTS: continue
    print(f"\n  attention geometry: {label}")
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**cfg["kwargs"])
    model.eval().float()
    n_prefix=int(getattr(model,"num_prefix_tokens",1))
    Hg,Wg=forward_grid(model,processor,source,BASE_RES); T=Hg*Wg
    blocks=get_vit_blocks(model,source); nh=RESULTS[label]["n_heads"]
    attns=[get_block_attention(b,source) for b in blocks]
    gate("part E: every block exposes a fused qkv",label,
         all(getattr(a,"qkv",None) is not None for a in attns),
         f"{sum(getattr(a,'qkv',None) is not None for a in attns)}/{len(attns)}")

    # exact gate: rebuilt q,k,v through the module's own fused path must equal its output
    cap={}; gh=[]
    for i,a in enumerate(attns):
        gh.append(a.register_forward_pre_hook(
            lambda m,args,kw,i=i: cap.__setitem__(("in",i),(args[0].detach(),kw.get("rope"))),with_kwargs=True))
        gh.append(a.register_forward_hook(lambda m,inp,out,i=i: cap.__setitem__(("out",i),out.detach())))
    torch.manual_seed(TORCH_SEED)
    dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=ATTN_BATCH,
                 batch_size=ATTN_BATCH,half=False,num_workers=0)
    with torch.no_grad():
        for im,_ in dl: model(im.to(DEVICE)); break
    for h in gh: h.remove()
    worst=0.0
    with torch.no_grad():
        for i,a in enumerate(attns):
            x,rope=cap[("in",i)]; out=cap[("out",i)]; B,N,C=x.shape
            npt=int(getattr(a,"num_prefix_tokens",n_prefix))
            q,k,v=attn_qkv(a,x,rope,npt)
            y=a.norm(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(B,N,C))
            g=getattr(a,"gate",None)
            if g is not None: y=y*g(x).sigmoid()
            worst=max(worst,(a.proj(y)-out).abs().max().item())
    cap.clear()
    gate("part E: rebuilt q/k/v reproduce the module output through its own fused path",label,
         worst==0.0,f"max |delta| over {len(attns)} blocks = {worst:.2e}")

    coords=torch.stack(torch.meshgrid(torch.arange(Hg),torch.arange(Wg),indexing="ij"),-1).reshape(-1,2).float().to(DEVICE)
    Dm=torch.cdist(coords,coords)
    acc={i:{"dist":torch.zeros(nh,device=DEVICE),"ent":torch.zeros(nh,device=DEVICE),
            "pref":torch.zeros(nh,device=DEVICE),"n":0} for i in range(len(blocks))}
    hs=[]
    def mk(i):
        def pre(mod,args,kw):
            x=args[0].detach(); rope=kw.get("rope"); npt=int(getattr(mod,"num_prefix_tokens",n_prefix))
            with torch.no_grad():
                att=attn_maps(mod,x,rope,npt)                    # (B,nh,N,N), rows sum to 1 over all keys
                ap=att[:,:,npt:,npt:]                             # patch queries x patch keys
                pref=1.0-ap.sum(-1)                               # mass a patch query sends to prefix tokens
                ap=ap/ap.sum(-1,keepdim=True).clamp_min(1e-12)    # renormalise over patch keys
                acc[i]["dist"]+=(ap*Dm).sum(-1).mean(-1).sum(0)
                acc[i]["ent"] +=(-(ap.clamp_min(1e-12)*ap.clamp_min(1e-12).log()).sum(-1)).mean(-1).sum(0)
                acc[i]["pref"]+=pref.mean(-1).sum(0)
                acc[i]["n"]+=x.shape[0]
            return None
        return pre
    for i,a in enumerate(attns): hs.append(a.register_forward_pre_hook(mk(i),with_kwargs=True))
    try:
        torch.manual_seed(TORCH_SEED)
        dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=ATTN_IMAGES,
                     batch_size=ATTN_BATCH,half=False,num_workers=0)
        with torch.no_grad():
            for im,_ in dl: model(im.to(DEVICE))
    except Exception as ex:
        print("    attention capture failed:",type(ex).__name__,ex)
    finally:
        for h in hs: h.remove()
    ATTN[label]={str(i):{"distance":(v["dist"]/max(v["n"],1)).tolist(),
                         "entropy":(v["ent"]/max(v["n"],1)).tolist(),
                         "prefix_mass":(v["pref"]/max(v["n"],1)).tolist()}
                 for i,v in acc.items() if v["n"]>0}
    ATTN[label]["_meta"]={"images":ATTN_IMAGES,"resolution":BASE_RES,"grid":[Hg,Wg],
                          "rope_applied":True,"qkv_bias_handled":True,
                          "softmax":"over all keys; patch block renormalised for distance/entropy",
                          "rebuild_gate_max_delta":worst}
    print(f"    captured {len(ATTN[label])-1} blocks x {nh} heads  (rebuild gate max delta {worst:.2e})")
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()
  if ATTN: save("partE_attention",ATTN)

# ---- cell 23 ----
import pandas as pd
pd.set_option("display.width",250); pd.set_option("display.float_format",lambda v:f"{v:,.4f}")

def post_blocks(): return [b for b in range(POST[0]+1,N_BLOCKS)]
def pret(e,r,b,h):
    pb=e["conditions"]["baseline"]["probes"].get(str(b)); p=r.get("probes",{}).get(str(b))
    if not pb or not p: return float("nan")
    ch=p["chance"][h]; rng=pb[h]-ch
    return (p[h]-ch)/rng if rng>1e-6 else float("nan")

print("PART A: generality of the three load-bearing claims\n")
rows=[]
for lab,e in RESULTS.items():
    PL=e["probe_layer"]; A=[d["head"] for d in e["stageA_ranking"]]; C=[d["head"] for d in e["stageC_ranking"]]
    ra={h:i for i,h in enumerate(A)}; rc={h:i for i,h in enumerate(C)}
    from scipy.stats import spearmanr
    rho=spearmanr([ra[h] for h in range(e["n_heads"])],[rc[h] for h in range(e["n_heads"])]).statistic
    rets=[d["ssdc_retention_at_probe"] for d in e["stageA_ranking"]]
    rows.append({"model":lab,"heads":e["n_heads"],
                 "axis_row_ret":sret(e,e["conditions"]["axis_row"]["ssdc"],PL),
                 "axis_col_ret":sret(e,e["conditions"]["axis_col"]["ssdc"],PL),
                 "prefix5_b11_ret":sret(e,e["conditions"]["Cref_prefix5"]["ssdc"],11),
                 "stageA_spread":max(rets)-min(rets),"top_head_ret":min(rets),
                 "A_to_C_rho":rho,"top_regen":C[0],"top_regen_A_rank":ra[C[0]]+1})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  carriers != regenerators replicates wherever A_to_C_rho is near zero")
print("  and top_regen_A_rank is far from 1.")

# ---- cell 24 ----
print("\nSTAGE D: axis specialisation, pre-registered as the MEAN over blocks 7-11\n")
for lab,e in RESULTS.items():
    print(f"--- {lab}")
    print(f"  {'head':>5}{'window':>8}{'row drop':>18}{'col drop':>18}   label")
    for d in e["stageC_ranking"][:e["d_top"]]:
        h=d["head"]
        if f"D_row_head{h:02d}" not in e["conditions"]: continue
        rr=e["conditions"]["Dref_axisrow5"]["probes"]; rc=e["conditions"]["Dref_axiscol5"]["probes"]
        pr=e["conditions"][f"D_row_head{h:02d}"]["probes"]; pc=e["conditions"][f"D_col_head{h:02d}"]["probes"]
        bl=post_blocks()
        rw_r=np.mean([(rr[str(b)]["row"]-pr[str(b)]["row"])*100 for b in bl])
        rw_c=np.mean([(rr[str(b)]["col"]-pr[str(b)]["col"])*100 for b in bl])
        cw_r=np.mean([(rc[str(b)]["row"]-pc[str(b)]["row"])*100 for b in bl])
        cw_c=np.mean([(rc[str(b)]["col"]-pc[str(b)]["col"])*100 for b in bl])
        lab_row = "ROW" if rw_r>2*max(rw_c,0.01) else ("COL" if cw_c>2*max(cw_r,0.01) else "mixed")
        print(f"  {h:>5}{'row':>8}{rw_r:>18.2f}{rw_c:>18.2f}   {lab_row}")
        print(f"  {'':>5}{'col':>8}{cw_r:>18.2f}{cw_c:>18.2f}")
    print()

# ---- cell 25 ----
if RES:
    print("\nPART C: does the mechanism carry extrapolation?\n")
    for lab,R in RES.items():
        print(f"--- {lab}   regenerator head {R['top_regenerator']}, random head {R['random_head']}")
        print(f"  {'res':>5}{'tokens':>8}{'base ssdc':>11}{'regenKO':>10}{'randKO':>10}"
              f"{'base top1':>11}{'regenKO top1':>14}{'randKO top1':>13}")
        for res,blk in R["by_res"].items():
            c=blk["conditions"]; PL=MODELS[lab]["probe_layer"]
            g=lambda n,k: c[n].get("acc",{}).get(k,float("nan"))
            print(f"  {res:>5}{blk['tokens']:>8}{c['baseline']['ssdc'][PL]:>11.4f}"
                  f"{c['regen_ko']['ssdc'][PL]:>10.4f}{c['random_ko']['ssdc'][PL]:>10.4f}"
                  f"{g('baseline','clean'):>11.4f}{g('regen_ko','clean'):>14.4f}{g('random_ko','clean'):>13.4f}")
      
        try:
            d={}
            for res,blk in R["by_res"].items():
                c=blk["conditions"]
                d[res]=(c["baseline"]["acc"]["clean"]-c["regen_ko"]["acc"]["clean"],
                        c["baseline"]["acc"]["clean"]-c["random_ko"]["acc"]["clean"])
            k=sorted(d,key=int)
            print(f"  top-1 cost of the regenerator knockout: {d[k[0]][0]*100:+.2f} pts at {k[0]}, "
                  f"{d[k[-1]][0]*100:+.2f} pts at {k[-1]}")
            print(f"  same for the random head:               {d[k[0]][1]*100:+.2f} pts at {k[0]}, "
                  f"{d[k[-1]][1]*100:+.2f} pts at {k[-1]}")
            print("  -> if the regenerator cost grows with resolution and the random one does not,")
            print("     the re-injection machinery is what carries extrapolation.")
        except Exception: pass
        print()

# ---- cell 26 ----
print("\nPART B: probe error bars (image bootstrap + probe seeds)\n")
noise_rows=[]
for lab,e in RESULTS.items():
    for cn in NOISE_CONDS:
        r=e["conditions"].get(cn)
        if not r: continue
        for b,p in r.get("probes",{}).items():
            if "noise" not in p: continue
            for head,n in p["noise"].items():
                noise_rows.append({"model":lab,"condition":cn,"block":int(b),"head":head,
                             "acc":p[head]*100,"boot_sd":n["boot_sd"]*100,
                             "seed_sd":n["seed_sd"]*100,"combined_sd":n["combined_sd"]*100})
if noise_rows:
    nd=pd.DataFrame(noise_rows)
    print(nd.groupby(["model","head"])[["boot_sd","seed_sd","combined_sd"]].mean().round(3).to_string())
    print("\n  The seed-only lower bound from the W3/W4 pairing was 0.15 points per probe.")
    print("  The bootstrap column is the image term that estimate was missing.")
else:
    print("  (part B not run)")

# ---- cell 28 ----
gdf=pd.DataFrame(GATES)
print("GATE REPORT\n"); print(gdf.to_string(index=False) if len(gdf) else "(none)")
n_fail=int((gdf.status=="FAIL").sum()) if len(gdf) else 0
# pre-model: 1 metric + 7 per layout x 2 + 2 composite = 17
# per model: grid, rope-rows, plane-decisive, axis-balanced, floor<baseline, block-0,
#            all-heads-vs-floor, hooks = 8  (anchored models add 2 + 3 more)
pre=17
per_model=8
expected=pre+per_model*len([l for l in RESULTS])
acct=len(GATES)>=expected
print(f"\ngates run {len(GATES)} | expected at least {expected} | accounting "
      f"{'OK' if acct else 'SHORT'}")
print("="*78)
if n_fail==0 and acct:
    print("ALL GATES PASSED")
else:
    if n_fail: print(f"{n_fail} GATE(S) FAILED")
    if not acct: print("GATE ACCOUNTING SHORT: some gates did not run")
    for _,r in gdf[gdf.status=="FAIL"].iterrows():
        print(f"   - [{r['scope']}] {r['gate']}: {r['detail']}")
print("="*78)

# ---- cell 30 ----
export={
 "notebook":"W5","generated_utc":RUN_STAMP,"parts_run":PARTS,
 "environment":{"torch":torch.__version__,"timm":timm.__version__,"device":DEVICE,
                "dtype":"float32","repo_path":str(repo),
                "gpu":torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"},
 "measurement":{"implementation":"repo metrics/ssdc.py evaluate_ssdc","RPI":True,
                "number_images":NUMBER_IMAGES,"batch_size":BATCH_SIZE,"metric":METRIC,
                "num_workers":0,"torch_seed":TORCH_SEED,"head_seed":HEAD_SEED,
                "base_res":BASE_RES,"extrap_res":EXTRAP_RES,
                "precision":"fp32; baseline and floor re-measured in-run per model per resolution"},
 "probe_settings":{"lr":PROBE_LR,"passes":PROBE_PASSES,"batch":PROBE_BATCH,
                   "train_frac":PROBE_TRAIN_FRAC,"split":"by image","standardised":True,
                   "seeding":"torch.manual_seed(TORCH_SEED + block), inside the probe loop",
                   "note":"W4 probe anchors will NOT reproduce under this seeding, by design",
                   "noise_seeds":NOISE_SEEDS,"noise_bootstrap":NOISE_BOOT},
 "comparison_policy":"intra-model only",
 "design_notes":{"window":WINDOW,"post":POST,
                 "stage_D_readout":D_READOUT,
                 "stage_D_top_n":{"12head":D_TOP_12HEAD,"6head":D_TOP_6HEAD},
                 "readout_offset":"first recoverable block after a window ending at m is m+2",
                 "rope_mixed_excluded":"learned frequencies are (2,12,12,32) and most planes are "
                                       "oblique; the row/col plane partition does not exist for it",
                 "reg1_gap_resolution":"run at 224 like the rest for comparability; native is 256"},
 "ssdc_anchors":ANCHORS,
 "results":RESULTS,"resolution":RES,"attention":ATTN,
 "gates":GATES,
 "gate_summary":{"total":len(GATES),"pass":int(sum(g["status"]=="PASS" for g in GATES)),
                 "fail":n_fail,"expected_at_least":expected,"accounting_ok":bool(acct)},
 "normalisation_protocol":{"formula":"(observed_b - floor_b) / (baseline_b - floor_b), per block",
                           "source":"in-run fp32 baseline and floor"},
}
out=OUT_DIR/f"w5_all_{RUN_STAMP.replace(':','-')}.json"
out.write_text(json.dumps(export,indent=2))

srows=[]
for lab,e in RESULTS.items():
    PL=e["probe_layer"]
    for n,r in e["conditions"].items():
        srows.append({"model":lab,"condition":n,"kind":r["kind"],
                      "heads_masked":str(r.get("heads_masked","")),
                      "ssdc_probe":r["ssdc"][PL],"ssdc_b11":r["ssdc"][11],
                      "ret_probe":sret(e,r["ssdc"],PL),"ret_b11":sret(e,r["ssdc"],11),
                      "hooks":r.get("hooks_registered"),"runtime_s":r.get("runtime_s")})
pd.DataFrame(srows).to_csv(OUT_DIR/"w5_summary.csv",index=False)
gdf.to_csv(OUT_DIR/"w5_gate_report.csv",index=False)
if noise_rows: pd.DataFrame(noise_rows).to_csv(OUT_DIR/"w5_probe_noise.csv",index=False)
rr=[]
for lab,R in RES.items():
    for res,blk in R["by_res"].items():
        for n,c in blk["conditions"].items():
            rr.append({"model":lab,"res":int(res),"tokens":blk["tokens"],"condition":n,
                       "ssdc_probe":c["ssdc"][MODELS[lab]["probe_layer"]],"ssdc_b11":c["ssdc"][11],
                       "clean_top1":c.get("acc",{}).get("clean"),"rpi_top1":c.get("acc",{}).get("rpi")})
if rr: pd.DataFrame(rr).to_csv(OUT_DIR/"w5_resolution.csv",index=False)

import shutil
archive=shutil.make_archive("w5_output","zip",OUT_DIR)
print("written:")
for p in sorted(OUT_DIR.iterdir()): print(f"  {p}  ({p.stat().st_size:,} bytes)")
print(f"\narchive: {archive} ({os.path.getsize(archive):,} bytes)")
try:
    from google.colab import files as cf; cf.download(archive)
except Exception: print("(not Colab: copy the archive back manually)")