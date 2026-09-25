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
import os, json, math, time, warnings
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

PARTS = []

N_IMAGES_ACC  = 5000        # part A accuracy
N_IMAGES_SSDC = 1000        # everywhere else, and part A's SSDC column
BATCH_SIZE    = 256
METRIC        = "manhattan"
TORCH_SEED    = 20260908
BASE_RES, EXTRAP_RES = 224, 320
ACC_ORDERS    = ["clean"]   # add "rpi" to also measure RPI-order top-1 (doubles part A)
WITH_SSDC_A   = True

WINDOW = list(range(0,6)); POST = list(range(6,12))
N_RANDOM_HEADS = 3          # part A, drawn from OUTSIDE the stage C top-D_TOP
RANDOM_HEAD_SEED = 90613    # not W5's HEAD_SEED (4127), which drew head 0 on every model

# part B, E1.2-style SSDC noise
NOISE_B_BLOCKS = None       # None -> [probe_layer, 11]; every stage C ranking is decided at 11
NOISE_B_BOOT   = 500
NOISE_B_SEEDS  = [20260908, 12345, 777, 2026]

# part C, probe noise
NOISE_C_CONDS  = ["baseline","Dref_axisrow5","Dref_axiscol5"]
NOISE_C_BLOCKS = [7,11]
PROBE_NOISE_SEEDS = [0,1,2]
PROBE_NOISE_BOOT  = 200

# part D, stage B random-k spread
D_KS = [2,4]; D_DRAWS = 3; D_SEED = 5150

PROBE_LR, PROBE_PASSES, PROBE_BATCH, PROBE_TRAIN_FRAC = 5e-3, 20, 512, 0.8
PROBE_CHUNK = 8192; PROBE_DEVICE = None

ANCHOR_TOL = 0.01
EQUIV_TOL  = 1e-9

# --- W5 results used as inputs. stageA for rope_ape is the both_floor ranking. ---
W5 = {
 "naver":    dict(kwargs=dict(), n_heads=12, d_top=4, probe_layer=4,
                  stageA=[2,9,11,10,5,0,3,7,8,1,6,4], stageC=[6,2,1,11,5,7,9,8,0,4,10,3],
                  baseline=0.469422, floor=0.012865, e12_sd=0.00834),
 "DINOv3":   dict(kwargs=dict(model_name="vit_base_patch16_dinov3.lvd1689m"), n_heads=12, d_top=4,
                  probe_layer=4, stageA=[2,6,5,7,4,0,11,8,10,1,9,3], stageC=[9,1,2,10,4,5,0,8,6,3,11,7],
                  baseline=0.948654, floor=0.017712, e12_sd=0.00077),
 "small":    dict(kwargs=dict(model_name="vit_small_patch16_rope_224.naver_in1k"), n_heads=6, d_top=3,
                  probe_layer=4, stageA=[4,3,0,5,2,1], stageC=[2,4,0,1,5,3],
                  baseline=0.371434, floor=0.011330, e12_sd=None),
 "rope_ape": dict(kwargs=dict(model_name="vit_base_patch16_rope_ape_224.naver_in1k"), n_heads=12, d_top=4,
                  probe_layer=4, stageA=[7,10,6,0,8,1,5,2,4,3,9,11], stageC=[3,9,11,7,2,5,10,6,1,8,0,4],
                  baseline=0.589777, floor=0.633780, e12_sd=None),
 "reg1_gap": dict(kwargs=dict(model_name="vit_base_patch16_rope_reg1_gap_256.sbb_in1k"), n_heads=12, d_top=4,
                  probe_layer=4, stageA=[0,5,4,1,6,8,7,2,10,11,9,3], stageC=[7,6,9,3,2,11,5,0,1,8,4,10],
                  baseline=0.689023, floor=0.014417, e12_sd=None),
}
# rope_ape is excluded from part A: its head knockouts cost ~0 top-1 (W5), and APE carries
# position at the probe layer, so it has no head-level accuracy claim to sharpen.
# DINOv3 is excluded from part A: num_classes = 0, so it has no classifier.
PART_A_MODELS = ["naver","reg1_gap","small"]
PART_B_MODELS = ["small","rope_ape","reg1_gap"]   # the three with no E1.2 noise scale
PART_C_MODELS = ["small","reg1_gap"]              # the two whose stage D labels use a proxy SD
PART_D_MODELS = ["naver","DINOv3"]

OUT_DIR=Path(os.environ.get("W7_OUT","w7_output")); OUT_DIR.mkdir(exist_ok=True)
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
def save(tag,obj):
    p=OUT_DIR/f"w7rope_{tag}_{RUN_STAMP.replace(':','-')}.json"
    p.write_text(json.dumps(obj,indent=2)); print(f"    saved -> {p.name}")

print(f"device {DEVICE} | parts {PARTS} | acc orders {ACC_ORDERS}")
print(f"accuracy at {N_IMAGES_ACC} images, everything else at {N_IMAGES_SSDC}")
print(f"run {RUN_STAMP}")

# ---- cell 7 ----
def resolve_repo():
    c=[]
    if REPO_PATH: c.append(Path(REPO_PATH).expanduser().resolve())
    cl=Path("vit-sae-analysis"); c+=[cl/"project_code"/"src", cl/"src", cl]
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
from scipy.stats import spearmanr

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

def forward_grid(model,processor,source):
    g={}
    conv=get_patch_embed_conv(model,source)
    h=conv.register_forward_hook(lambda m,i,o: g.update(hw=(int(o.shape[2]),int(o.shape[3])) if o.dim()==4 else None))
    dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=BATCH_SIZE,
                 batch_size=BATCH_SIZE,half=False,num_workers=0)
    with torch.no_grad():
        for im,_ in dl: model(im.to(DEVICE)); break
    h.remove(); return g.get("hw")

def attn_attrs(model,source):
    rh,src,nh=False,"default(interleaved)",None
    for blk in get_vit_blocks(model,source):
        a=get_block_attention(blk,source)
        for at in ("rotate_half","half","rope_rotate_half"):
            v=getattr(a,at,None)
            if isinstance(v,bool): rh,src=v,at; break
        nh=getattr(a,"num_heads",None); break
    return rh,src,nh

def plane_columns(p,n,rh): return [p,p+n] if rh else [2*p,2*p+1]
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

class NullCtx:
    handles=[]
    def __enter__(s): return s
    def __exit__(s,*e): return False

def ssdc_only(model,processor,source,ctx,n_images,seed=TORCH_SEED):
    torch.manual_seed(seed); np.random.seed(seed%(2**31))
    with ctx:
        hooks=len(getattr(ctx,"handles",[]) or [])
        scores,_=evaluate_ssdc(model,processor,DATASET,source,RPI=True,number_images=n_images,
                               batch_size=BATCH_SIZE,metric=METRIC,half=False,num_workers=0)
    return [float(s) for s in scores],hooks

def accuracy(model,processor,source,ctx,n_images,orders=None):
    out={}
    for tag in (orders or ACC_ORDERS):
        torch.manual_seed(TORCH_SEED)
        dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=n_images,
                     batch_size=BATCH_SIZE,half=False,num_workers=0)
        with ctx: out[tag]=float(predict(model,dl,source,RPI=(tag=="rpi"),half=False))
    return out

# predict() plus per-image correctness, for paired SEs across arms scored on the same images.
# Matches predict() exactly on both RPI orders. Per-image arrays are not written to the JSON.
def predict_paired(model,dataloader,source,RPI=False,half=False):
    device="cuda" if torch.cuda.is_available() else "cpu"
    handle=None
    if RPI:
        def RPI_hook(module,inp,output):
            B,C,H,W=output.shape; out=output.view(B,C,-1).contiguous()
            perm=torch.randperm(out.shape[-1])
            return out[:,:,perm].view(B,C,H,W)
        handle=get_patch_embed_conv(model,source).register_forward_hook(RPI_hook)
    model.eval(); correct=[]
    with torch.inference_mode():
        for images,labels in dataloader:
            images=images.to(device)
            logits=model(images) if source=="timm" else model(**images).logits
            pred=logits.argmax(-1).to(device)
            correct.append((pred==torch.as_tensor(labels).to(device)).cpu())
    if RPI: handle.remove()
    correct=torch.cat(correct)
    return float(correct.float().mean()),correct

def accuracy_paired(model,processor,source,ctx,n_images,orders=None):
    out={}; corr={}
    for tag in (orders or ACC_ORDERS):
        torch.manual_seed(TORCH_SEED)
        dl=prep_data(DATASET,processor,source,corruption_type=None,number_images=n_images,
                     batch_size=BATCH_SIZE,half=False,num_workers=0)
        with ctx: m,c=predict_paired(model,dl,source,RPI=(tag=="rpi"),half=False)
        out[tag]=m; corr[tag]=c
    return out,corr

def paired_se(a,b):
    """SE of mean(a-b) for two same-length 0/1 correctness tensors on the same images."""
    d=(a.float()-b.float())
    n=d.shape[0]
    return float(d.mean().item()),float(d.std(unbiased=True).item()/(n**0.5))

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
    full=_apply(x,e,rh); iden=_apply(x,identity_like(e),rh)
    gate(f"mask NO heads equals unmodified rope [{lname}]","per-head",
         torch.allclose(_apply(x,per_head([],identity_like,H)(e),rh),full,atol=1e-6))
    gate(f"mask ALL heads equals global identity [{lname}]","per-head",
         torch.allclose(_apply(x,per_head(range(H),identity_like,H)(e),rh),iden,atol=1e-6))
    one=_apply(x,per_head([3],identity_like,H)(e),rh); others=[h for h in range(H) if h!=3]
    gate(f"mask head 3 leaves others untouched [{lname}]","per-head",
         torch.allclose(one[:,others],full[:,others],atol=1e-6))

class _Stub:
    def __init__(s,n=12): s.blocks=[nn.Identity() for _ in range(n)]
try:
    RopeComposite(_Stub(),"timm",[(range(0,6),identity_like),(range(5,12),identity_like)]); ov=False
except ValueError as ex: ov="overlapping" in str(ex)
gate("composite REJECTS overlapping segments","composite",ov)

# ---- cell 13 ----
DRAWS={}
for nm in PART_A_MODELS:
    w=W5[nm]; nh=w["n_heads"]; top=set(w["stageC"][:w["d_top"]])
    cands=[h for h in range(nh) if h not in top]
    rng=np.random.default_rng(RANDOM_HEAD_SEED+sum(ord(ch) for ch in nm))
    n_draw=min(N_RANDOM_HEADS,len(cands))
    heads=sorted(rng.choice(cands,size=n_draw,replace=False).tolist())
    ra={h:i+1 for i,h in enumerate(w["stageA"])}; rc={h:i+1 for i,h in enumerate(w["stageC"])}
    DRAWS[nm]={"random_heads":heads,"candidates":cands,"exhaustive":n_draw==len(cands),
               "ranks":{str(h):{"A":ra[h],"C":rc[h]} for h in heads},
               "regen_rank1":w["stageC"][0],"regen_rank2":w["stageC"][1],
               "regen_rank1_A":ra[w["stageC"][0]],"regen_rank2_A":ra[w["stageC"][1]]}
    print(f"{nm:10s} regen r1=h{w['stageC'][0]} (A rank {ra[w['stageC'][0]]})  "
          f"r2=h{w['stageC'][1]} (A rank {ra[w['stageC'][1]]})")
    print(f"{'':10s} random {heads} -> ranks "
          f"{ {h:(ra[h],rc[h]) for h in heads} }   pool {len(cands)}"
          + ("  (exhaustive)" if n_draw==len(cands) else ""))
    bad=[h for h in heads if h in top]
    gate(f"{nm}: random heads are outside the stage C top-{w['d_top']}","draw",not bad,
         f"offenders {bad}" if bad else f"heads {heads}")
save("draws",DRAWS)


# =============================================================================
# W7 (reviewer controls, RoPE side). Built on W6's helpers above, unchanged.
#   R1  B2+B4: every head's rotations off in blocks 6-11, intact AND after the 0-5
#       window; top-1 (5,000 images, per-image correctness saved) + SSDC per block.
#   R2  B3: shuffled rotations (rotations of a fixed random permutation of the patch
#       positions: same statistics, wrong position) vs identity rotations.
#   R3  C3: (a) probes at blocks 4 and 11 with a train/val/test split (epoch chosen on
#       val), per-row / per-column accuracy vs distance to the border; (b) the same with
#       patch->prefix attention blocked in every block.
# =============================================================================
import traceback
PERIMG_DIR=OUT_DIR/"perimage"; PERIMG_DIR.mkdir(exist_ok=True, parents=True)
R_MODELS=os.environ.get("W7_MODELS","naver,reg1_gap,small,rope_ape").split(",")
C3_MODELS=os.environ.get("W7_C3_MODELS","naver,reg1_gap,small,DINOv3").split(",")
N_ACC=int(os.environ.get("W7_N_ACC",N_IMAGES_ACC)); N_SS=int(os.environ.get("W7_N_SS",N_IMAGES_SSDC))
N_PROBE=int(os.environ.get("W7_N_PROBE",1000))
IDEN=identity_like

def shuffled_like(perm):
    def tf(emb):
        if emb.dim()!=2: raise RuntimeError(f"shuffle expects 2-D rope, got {tuple(emb.shape)}")
        return emb[perm]
    return tf

def run_cond(model,processor,source,ctx_fn,with_acc=True):
    rec={}
    ss,hooks=ssdc_only(model,processor,source,ctx_fn(),N_SS); rec["ssdc"]=ss; rec["hooks"]=hooks
    corr=None
    if with_acc:
        a,c=accuracy_paired(model,processor,source,ctx_fn(),N_ACC); rec["acc"]=a["clean"]; corr=c["clean"]
    return rec,corr

def part_R12(nm):
    w=W5[nm]; out={"model":nm,"conditions":{}}; t0=time.time()
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**w["kwargs"])
    model.eval().float()
    rh,_,nh_a=attn_attrs(model,source); nh=int(nh_a) if nh_a else w["n_heads"]
    has_head=getattr(model,"num_classes",1)>0
    T=196; g=torch.Generator().manual_seed(7)
    perms=[torch.randperm(T,generator=g) for _ in range(2)]
    r1=w["stageC"][0]
    conds=[("baseline",lambda:NullCtx()),
           ("floor_identity",lambda:RopeGlobal(model,IDEN)),
           ("floor_shuffled_s0",lambda:RopeGlobal(model,shuffled_like(perms[0]))),
           ("floor_shuffled_s1",lambda:RopeGlobal(model,shuffled_like(perms[1]))),
           ("window_identity",lambda:RopeComposite(model,source,[(WINDOW,IDEN)])),
           ("window_shuffled",lambda:RopeComposite(model,source,[(WINDOW,shuffled_like(perms[0]))]))]
    for h in range(nh):
        conds.append((f"win_h{h}",(lambda h=h:RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([h],IDEN,nh))]))))
        conds.append((f"int_h{h}",(lambda h=h:RopeComposite(model,source,[(POST,per_head([h],IDEN,nh))]))))
    conds.append((f"win_h{r1}_shuffled",lambda:RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([r1],shuffled_like(perms[0]),nh))])))
    conds.append((f"int_h{r1}_shuffled",lambda:RopeComposite(model,source,[(POST,per_head([r1],shuffled_like(perms[0]),nh))])))
    only=[c for c in os.environ.get("W7_ONLY","").split(",") if c]
    if only: conds=[c for c in conds if c[0] in only]
    corr_store={}
    for cname,cf in conds:
        t=time.time()
        try:
            rec,corr=run_cond(model,processor,source,cf,with_acc=has_head)
            if corr is not None: corr_store[cname]=corr.numpy().astype(np.uint8)
            rec["runtime_s"]=round(time.time()-t,1); out["conditions"][cname]=rec
            print(f"  {nm} {cname:22s} ssdc4={rec['ssdc'][4]:+.4f} b11={rec['ssdc'][11]:+.4f} "
                  +(f"acc={rec['acc']*100:.2f}" if 'acc' in rec else "")+f"  {rec['runtime_s']}s",flush=True)
        except Exception as ex:
            out["conditions"][cname]={"error":repr(ex)}; print("  ERROR",cname,repr(ex)); traceback.print_exc()
        save(f"R12_{nm}",out)
        if corr_store: np.savez_compressed(PERIMG_DIR/f"R12_{nm}_correct.npz",**corr_store)
    out["n_heads"]=nh; out["top_rebuilder"]=r1; out["runtime_s"]=round(time.time()-t0,1)
    save(f"R12_{nm}",out)
    del model; torch.cuda.empty_cache()

# ---------------- probes with a validation split (B8 fix, C3) ----------------
class CapIn:
    def __init__(s,model,source,blocks,npt):
        s.b=get_vit_blocks(model,source); s.which=list(blocks); s.npt=int(npt); s.source=source
        s.store={i:[] for i in s.which}; s.h=[]
    def __enter__(s):
        for i in s.which:
            def hook(m,inp,out,i=i):
                t=inp[0].detach()
                if s.npt>0: t=t[:,s.npt:,:]
                s.store[i].append(t.half().cpu())
            s.h.append(get_block_attention(s.b[i],s.source).register_forward_hook(hook))
        return s
    def __exit__(s,*e):
        for x in s.h: x.remove()
        s.h=[]

def fit_probe_val(X,y,nc,n_img,T,seed=0):
    g=torch.Generator().manual_seed(seed); idx=torch.randperm(n_img,generator=g)
    ntr,nva=int(0.6*n_img),int(0.2*n_img)
    rows=lambda ids:(ids.unsqueeze(1)*T+torch.arange(T).unsqueeze(0)).reshape(-1)
    tr,va,te=rows(idx[:ntr]),rows(idx[ntr:ntr+nva]),rows(idx[ntr+nva:])
    Xg=X.to(DEVICE).float(); yg=y.to(DEVICE)
    mu=Xg[tr].mean(0,keepdim=True); sd=Xg[tr].std(0,keepdim=True).clamp_min(1e-6); Xg=(Xg-mu)/sd
    torch.manual_seed(seed); head=nn.Linear(Xg.shape[1],nc).to(DEVICE)
    opt=torch.optim.AdamW(head.parameters(),lr=PROBE_LR)
    best_va=-1; te_at_best=None; pred_at_best=None; peak_te=0; hist=[]
    for ep in range(PROBE_PASSES):
        p=tr[torch.randperm(len(tr))]; head.train()
        for s0 in range(0,len(p),PROBE_BATCH):
            b=p[s0:s0+PROBE_BATCH].to(DEVICE); opt.zero_grad(); F.cross_entropy(head(Xg[b]),yg[b]).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv=torch.cat([head(Xg[va[s:s+8192].to(DEVICE)]).argmax(-1) for s in range(0,len(va),8192)])
            pt=torch.cat([head(Xg[te[s:s+8192].to(DEVICE)]).argmax(-1) for s in range(0,len(te),8192)])
        av=(pv==yg[va.to(DEVICE)]).float().mean().item(); at=(pt==yg[te.to(DEVICE)]).float().mean().item()
        hist.append([av,at]); peak_te=max(peak_te,at)
        if av>best_va: best_va=av; te_at_best=at; pred_at_best=pt.cpu()
    del Xg,head,opt; torch.cuda.empty_cache()
    return {"test_at_best_val":te_at_best,"final_test":hist[-1][1],"peak_test_old_protocol":peak_te,
            "best_val":best_va,"history":hist},pred_at_best,te

def probe_battery(model,processor,source,ctx_fn,blocks,npt,H=14,W=14):
    cap=CapIn(model,source,blocks,npt)
    with ctx_fn():
        with cap:
            evaluate_ssdc(model,processor,DATASET,source,RPI=True,number_images=N_PROBE,
                          batch_size=BATCH_SIZE,metric=METRIC,half=False,num_workers=0)
    res={}
    for b in blocks:
        X=torch.cat(cap.store[b],0); n_img,T,D=X.shape; X=X.reshape(-1,D)
        pos=torch.arange(T).repeat(n_img); row=pos//W; col=pos%W
        rb={}
        for name,y,nc in [("exact",pos,T),("row",row,H),("col",col,W)]:
            r,pred,te=fit_probe_val(X,y,nc,n_img,T)
            if name in ("row","col"):
                yt=y[te]; per=[]
                for k in range(nc):
                    m=(yt==k); per.append(float((pred[m]==k).float().mean()) if m.any() else None)
                r["per_index_acc"]=per
                dist=torch.minimum(yt,(nc-1)-yt)
                r["acc_by_border_distance"]={int(d):float((pred[dist==d]==yt[dist==d]).float().mean()) for d in dist.unique()}
            del r["history"]; rb[name]=r
        res[str(b)]=rb
    return res

class PrefixBlock:
    """Block patch-query -> prefix-key attention in every block via an additive attn_mask."""
    def __init__(s,model,source,npt,T):
        s.b=get_vit_blocks(model,source); s.source=source; s.npt=npt; s.N=npt+T; s.h=[]
    def __enter__(s):
        N,npt=s.N,s.npt
        def pre(mod,args,kw):
            x=args[0] if args else kw.get("x")
            m=torch.zeros(1,1,N,N,device=x.device,dtype=x.dtype)
            m[...,npt:,:npt]=float("-inf")
            kw=dict(kw); old=kw.get("attn_mask",None)
            kw["attn_mask"]=m if old is None else old+m
            return args,kw
        for blk in s.b:
            s.h.append(get_block_attention(blk,s.source).register_forward_pre_hook(pre,with_kwargs=True))
        return s
    def __exit__(s,*e):
        for x in s.h: x.remove()
        s.h=[]

def part_C3(nm):
    w=W5[nm]; out={"model":nm}; t0=time.time()
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**w["kwargs"])
    model.eval().float()
    npt=int(getattr(model,"num_prefix_tokens",1)); out["num_prefix_tokens"]=npt
    blocks=[4,11]
    try:
        out["intact"]=probe_battery(model,processor,source,lambda:NullCtx(),blocks,npt); save(f"C3_{nm}",out)
    except Exception as ex: out["intact"]={"error":repr(ex)}; traceback.print_exc()
    try:
        ss,_=ssdc_only(model,processor,source,PrefixBlock(model,source,npt,196),N_SS); out["prefix_block_ssdc"]=ss
        out["prefix_block"]=probe_battery(model,processor,source,lambda:PrefixBlock(model,source,npt,196),blocks,npt)
        if getattr(model,"num_classes",1)>0:
            a,_=accuracy_paired(model,processor,source,PrefixBlock(model,source,npt,196),N_ACC); out["prefix_block_acc"]=a["clean"]
    except Exception as ex: out["prefix_block"]={"error":repr(ex)}; traceback.print_exc()
    out["runtime_s"]=round(time.time()-t0,1); save(f"C3_{nm}",out)
    del model; torch.cuda.empty_cache()

for nm in R_MODELS:
    print("\n=== R12",nm,"===",flush=True)
    try: part_R12(nm)
    except Exception as ex: print("R12 FAILED",nm,repr(ex)); traceback.print_exc()
for nm in C3_MODELS:
    print("\n=== C3",nm,"===",flush=True)
    try: part_C3(nm)
    except Exception as ex: print("C3 FAILED",nm,repr(ex)); traceback.print_exc()
save("gates",GATES)
print("W7 ROPE DONE",flush=True)
