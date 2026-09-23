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

PARTS = ["A","B","C","D"]

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

OUT_DIR=Path("w6_output"); OUT_DIR.mkdir(exist_ok=True)
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
    p=OUT_DIR/f"w6_{tag}_{RUN_STAMP.replace(':','-')}.json"
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

# ---- cell 15 ----
PART_A={}
if "A" in PARTS:
  for nm in PART_A_MODELS:
    w=W5[nm]; PL=w["probe_layer"]
    print("\n"+"="*78); print(f"PART A: {nm}"); print("="*78)
    t0=time.time()
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**w["kwargs"])
    model.eval().float()
    rh,rh_src,nh_a=attn_attrs(model,source); n_heads=int(nh_a) if nh_a else w["n_heads"]
    Hg,Wg=forward_grid(model,processor,source)
    gate(f"{nm}: num_heads matches W5","partA",n_heads==w["n_heads"],f"{n_heads} vs {w['n_heads']}")
    gate(f"{nm}: grid is 14x14 at {BASE_RES}","partA",(Hg,Wg)==(14,14),f"{Hg}x{Wg}")

    # --- anchor check at the W5 sample size ---------------------------------
    b_anchor,_=ssdc_only(model,processor,source,NullCtx(),N_IMAGES_SSDC)
    f_anchor,_=ssdc_only(model,processor,source,RopeGlobal(model,identity_like),N_IMAGES_SSDC)
    gate(f"{nm}: baseline reproduces W5 at {N_IMAGES_SSDC} images","partA",
         abs(b_anchor[PL]-w["baseline"])<ANCHOR_TOL,f"{b_anchor[PL]:.6f} vs {w['baseline']}")
    gate(f"{nm}: floor reproduces W5 at {N_IMAGES_SSDC} images","partA",
         abs(f_anchor[PL]-w["floor"])<ANCHOR_TOL,f"{f_anchor[PL]:.6f} vs {w['floor']}")
    rall,_=ssdc_only(model,processor,source,
                     RopeComposite(model,source,[(range(N_BLOCKS),per_head(range(n_heads),identity_like,n_heads))]),
                     N_IMAGES_SSDC)
    dmax=max(abs(a-b) for a,b in zip(rall,f_anchor))
    gate(f"{nm}: per-head mask of ALL heads reproduces the global floor","partA",dmax<EQUIV_TOL,f"{dmax:.2e}")
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()

    entry={"n_heads":n_heads,"probe_layer":PL,"rotate_half":bool(rh),
           "anchor_1000":{"baseline":b_anchor[PL],"floor":f_anchor[PL]},
           "draw":DRAWS[nm],"by_res":{}}
    r1,r2=DRAWS[nm]["regen_rank1"],DRAWS[nm]["regen_rank2"]

    for res in (BASE_RES,EXTRAP_RES):
        model,processor,source=load_rope(device=DEVICE,half=False,input_size=res,**w["kwargs"])
        model.eval().float()
        rh,_,nh_a=attn_attrs(model,source); n_heads=int(nh_a) if nh_a else w["n_heads"]
        Hg,Wg=forward_grid(model,processor,source)
        gate(f"{nm}: grid matches resolution {res}","partA",Hg==res//16 and Wg==res//16,f"{Hg}x{Wg}")
        print(f"  res {res}: grid {Hg}x{Wg} = {Hg*Wg} tokens")

        IDEN=identity_like
        conds=[("baseline",NullCtx()),
               ("floor",RopeGlobal(model,IDEN)),
               ("Cref",RopeComposite(model,source,[(WINDOW,IDEN)])),
               (f"regen_r1_h{r1}",RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([r1],IDEN,n_heads))])),
               (f"regen_r2_h{r2}",RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([r2],IDEN,n_heads))]))]
        for h in DRAWS[nm]["random_heads"]:
            conds.append((f"random_h{h}",
                          RopeComposite(model,source,[(WINDOW,IDEN),(POST,per_head([h],IDEN,n_heads))])))

        block={}; correctness={}
        for cname,ctx in conds:
            t=time.time(); rec={}
            if WITH_SSDC_A:
                ss,hooks=ssdc_only(model,processor,source,ctx,N_IMAGES_SSDC)
                rec["ssdc"]=ss; rec["hooks_registered"]=hooks
            rec["acc"],correctness[cname]=accuracy_paired(model,processor,source,ctx,N_IMAGES_ACC)
            rec["runtime_s"]=round(time.time()-t,1)
            block[cname]=rec
            a=rec["acc"]
            print(f"    {cname:16s} "
                  + (f"ssdc@{PL}={rec['ssdc'][PL]:+.4f} b11={rec['ssdc'][11]:+.4f}  " if WITH_SSDC_A else "")
                  + "  ".join(f"{k}={v*100:.2f}" for k,v in a.items())
                  + f"   {rec['runtime_s']}s")
        if WITH_SSDC_A:
            gate(f"{nm}: floor below baseline at {res}","partA",
                 block["floor"]["ssdc"][PL] < block["baseline"]["ssdc"][PL]-0.01)
            inert=[c for c in block if c!="baseline" and block[c].get("hooks_registered",0)==0]
            gate(f"{nm}: every intervening condition registered a hook at {res}","partA",not inert,
                 f"inert {inert}" if inert else "")

        # Paired SE of regen_r1/r2 against the pooled random draws and against each draw.
        rand_names=[f"random_h{h}" for h in DRAWS[nm]["random_heads"]]
        paired={}
        for order in ACC_ORDERS:
            cref_c=correctness["Cref"][order]
            rand_c=[correctness[rn][order] for rn in rand_names]
            rand_pool=torch.stack(rand_c).float().mean(0)  # per-image P(random head correct)
            for lbl in (f"regen_r1_h{r1}",f"regen_r2_h{r2}"):
                head_c=correctness[lbl][order]
                d_mean,d_se=paired_se(head_c,rand_pool)
                per_rand=[paired_se(head_c,rc) for rc in rand_c]
                paired[f"{lbl}__vs_random_pool__{order}"]={
                    "cost_vs_Cref_pts":round((cref_c.float().mean().item()-head_c.float().mean().item())*100,3),
                    "extra_cost_vs_random_pts":round(-d_mean*100,3),
                    "se_pts":round(d_se*100,3),
                    "per_random_extra_cost_pts":[round(-m*100,3) for m,_ in per_rand],
                    "per_random_se_pts":[round(s*100,3) for _,s in per_rand]}
                print(f"    {lbl} vs random pool ({order}): "
                      f"{paired[f'{lbl}__vs_random_pool__{order}']['extra_cost_vs_random_pts']:+.2f} "
                      f"+- {paired[f'{lbl}__vs_random_pool__{order}']['se_pts']:.2f} pts (paired)")
        entry["by_res"][str(res)]={"grid":[Hg,Wg],"tokens":Hg*Wg,"conditions":block,"paired":paired}
        del model
        if DEVICE=="cuda": torch.cuda.empty_cache()

    entry["runtime_s"]=round(time.time()-t0,1)
    PART_A[nm]=entry; save(f"partA_{nm}",entry)
    print(f"  {nm} done in {entry['runtime_s']}s")

# ---- cell 17 ----
class SimCapture:
    '''Per-image upper-triangle cosine similarity at chosen blocks, prefix stripped.'''
    def __init__(s,model,source,blocks,n_prefix,T):
        s.blocks=get_vit_blocks(model,source); s.which=list(blocks); s.n_prefix=int(n_prefix)
        s.source=source; s.store={i:[] for i in s.which}; s.handles=[]
        s.iu=torch.triu_indices(T,T,offset=1)
    def __enter__(s):
        def mk(i):
            def hook(mod,inp,out):
                t=inp[0].detach().float()
                if s.n_prefix>0: t=t[:,s.n_prefix:,:]
                t=t/t.norm(dim=-1,keepdim=True).clamp_min(1e-8)   # matches metrics/ssdc.py
                S=t@t.transpose(-2,-1)                       # (B,T,T)
                s.store[i].append(S[:,s.iu[0],s.iu[1]].cpu())
            return hook
        for i in s.which:
            s.handles.append(get_block_attention(s.blocks[i],s.source).register_forward_hook(mk(i)))
        return s
    def __exit__(s,*e):
        for h in s.handles: h.remove()
        s.handles=[]
    def stacked(s,i): return torch.cat(s.store[i],0) if s.store[i] else None

def ssdc_from_iu(mean_iu,H,W,neg_dist_iu):
    return float(spearmanr(neg_dist_iu,mean_iu).statistic)

PART_B={}
if "B" in PARTS:
  for nm in PART_B_MODELS:
    w=W5[nm]; PL=w["probe_layer"]
    print("\n"+"="*78); print(f"PART B: {nm}"); print("="*78)
    t0=time.time()
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**w["kwargs"])
    model.eval().float()
    n_prefix=int(getattr(model,"num_prefix_tokens",1))
    Hg,Wg=forward_grid(model,processor,source); T=Hg*Wg
    blocks=NOISE_B_BLOCKS or [PL,N_BLOCKS-1]
    coords=np.stack(np.meshgrid(np.arange(Hg),np.arange(Wg),indexing="ij"),-1).reshape(-1,2)
    Dm=_cdist(coords,coords,metric="cityblock")
    iu=np.triu_indices(T,k=1); neg_d=-Dm[iu]

    cap=SimCapture(model,source,blocks,n_prefix,T); cap.__enter__()
    try:
        ref,_=ssdc_only(model,processor,source,NullCtx(),N_IMAGES_SSDC,seed=NOISE_B_SEEDS[0])
    finally:
        cap.__exit__()

    res={}
    for b in blocks:
        M=cap.stacked(b)
        if M is None: continue
        n_img=M.shape[0]
        full=ssdc_from_iu(M.mean(0).numpy(),Hg,Wg,neg_d)
        gate(f"{nm}: cached recomputation matches the repo SSDC at block {b}","partB",
             abs(full-ref[b])<1e-5,f"{full:.8f} vs {ref[b]:.8f}  (delta {abs(full-ref[b]):.2e})")
        g=np.random.default_rng(TORCH_SEED+b)
        boot=[ssdc_from_iu(M[g.integers(0,n_img,n_img)].mean(0).numpy(),Hg,Wg,neg_d)
              for _ in range(NOISE_B_BOOT)]
        res[str(b)]={"point":full,"repo":ref[b],"n_images":int(n_img),
                     "boot_mean":float(np.mean(boot)),"boot_sd":float(np.std(boot,ddof=1))}
    cap.store.clear(); del cap

    seed_vals={str(b):[] for b in blocks}
    for sd in NOISE_B_SEEDS:
        ss,_=ssdc_only(model,processor,source,NullCtx(),N_IMAGES_SSDC,seed=sd)
        for b in blocks: seed_vals[str(b)].append(ss[b])
    for b in blocks:
        v=np.array(seed_vals[str(b)])
        res[str(b)]["seed_values"]=v.tolist()
        res[str(b)]["seed_sd"]=float(v.std(ddof=1))
        res[str(b)]["combined_sd"]=float(np.sqrt(res[str(b)]["boot_sd"]**2+v.std(ddof=1)**2))
        r=res[str(b)]
        print(f"  block {b:2d}: point {r['point']:.4f}  boot SD {r['boot_sd']:.5f}  "
              f"seed SD {r['seed_sd']:.5f}  combined {r['combined_sd']:.5f}")
    PART_B[nm]={"blocks":blocks,"by_block":res,"runtime_s":round(time.time()-t0,1)}
    save(f"partB_{nm}",PART_B[nm])
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()

# ---- cell 19 ----
class Capture:
    def __init__(s,model,source,blocks,n_prefix):
        s.blocks=get_vit_blocks(model,source); s.which=list(blocks); s.n_prefix=int(n_prefix)
        s.source=source; s.store={i:[] for i in s.which}; s.handles=[]
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
    return float(peak),best

def probe_block_noise(feats,H,W,n_images,block):
    T=H*W
    g=torch.Generator().manual_seed(TORCH_SEED)
    idx=torch.randperm(n_images,generator=g); n_tr=int(round(n_images*PROBE_TRAIN_FRAC))
    rows=lambda ids:(ids.unsqueeze(1)*T+torch.arange(T).unsqueeze(0)).reshape(-1)
    tr,te=rows(idx[:n_tr]),rows(idx[n_tr:]); n_te=n_images-n_tr
    Xtr=feats[tr].to(PDEV); Xte=feats[te].to(PDEV)
    mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True).clamp_min(1e-6)
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    slot=torch.arange(T); out={}
    for name,lab,nc in (("exact",slot,T),("row",slot//W,H),("col",slot%W,W)):
        y=lab.repeat(n_images)
        acc,preds=_fit(Xtr,y[tr].to(PDEV),Xte,y[te].to(PDEV),nc,TORCH_SEED+block)
        corr=(preds==y[te].to(PDEV)).float().reshape(n_te,T).mean(1).cpu().numpy()
        g2=np.random.default_rng(TORCH_SEED+block)
        boot=[corr[g2.integers(0,n_te,n_te)].mean() for _ in range(PROBE_NOISE_BOOT)]
        seeds=[_fit(Xtr,y[tr].to(PDEV),Xte,y[te].to(PDEV),nc,TORCH_SEED+block+1000*(s+1))[0]
               for s in PROBE_NOISE_SEEDS]
        out[name]={"acc":acc,"boot_sd":float(np.std(boot,ddof=1)),
                   "seed_sd":float(np.std(seeds,ddof=1)),"seed_values":seeds,
                   "combined_sd":float(np.sqrt(np.var(boot,ddof=1)+np.var(seeds,ddof=1)))}
    out["chance"]={"exact":1.0/T,"row":1.0/H,"col":1.0/W}
    del Xtr,Xte
    if PDEV=="cuda": torch.cuda.empty_cache()
    return out

PART_C={}
if "C" in PARTS:
  for nm in PART_C_MODELS:
    w=W5[nm]; PL=w["probe_layer"]
    print("\n"+"="*78); print(f"PART C: {nm}"); print("="*78)
    t0=time.time()
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**w["kwargs"])
    model.eval().float()
    rh,_,nh_a=attn_attrs(model,source); n_heads=int(nh_a) if nh_a else w["n_heads"]
    n_prefix=int(getattr(model,"num_prefix_tokens",1))
    Hg,Wg=forward_grid(model,processor,source)
    rname,rmod=find_rope(model); emb0=rmod.get_embed(shape=(Hg,Wg)); n_planes=int(emb0.shape[-1]//4)
    sin,cos=emb0.detach().float().chunk(2,-1); rowp=[];colp=[]
    for p in range(n_planes):
        c=plane_columns(p,n_planes,rh)[0]
        z=torch.complex(cos[:,c],sin[:,c]).reshape(Hg,Wg)
        ra=(z-z.mean(dim=1,keepdim=True)).abs().mean().item()
        rb=(z-z.mean(dim=0,keepdim=True)).abs().mean().item()
        (rowp if ra<rb else colp).append(p)
    gate(f"{nm}: axis split is balanced","partC",len(rowp)==len(colp)==n_planes//2,f"{len(rowp)}/{len(colp)}")
    ROWM=lambda x: mask_planes(x,rowp,n_planes,rh)
    COLM=lambda x: mask_planes(x,colp,n_planes,rh)
    ctxs={"baseline":lambda: NullCtx(),
          "Dref_axisrow5":lambda: RopeComposite(model,source,[(WINDOW,ROWM)]),
          "Dref_axiscol5":lambda: RopeComposite(model,source,[(WINDOW,COLM)])}
    out={}
    for cname in NOISE_C_CONDS:
        cap=Capture(model,source,NOISE_C_BLOCKS,n_prefix); cap.__enter__()
        try: ssdc_only(model,processor,source,ctxs[cname](),N_IMAGES_SSDC)
        finally: cap.__exit__()
        out[cname]={}
        for b in NOISE_C_BLOCKS:
            a=cap.stacked(b)
            if a is None: continue
            out[cname][str(b)]=probe_block_noise(a.reshape(-1,a.shape[-1]),Hg,Wg,a.shape[0],b)
            r=out[cname][str(b)]
            print(f"  {cname:16s} block {b:2d}: "
                  + "  ".join(f"{h} {r[h]['acc']*100:.2f}+-{r[h]['combined_sd']*100:.2f}"
                              for h in ("exact","row","col")))
            del a
        cap.store.clear()
    PART_C[nm]={"blocks":NOISE_C_BLOCKS,"conditions":out,"runtime_s":round(time.time()-t0,1)}
    save(f"partC_{nm}",PART_C[nm])
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()

# ---- cell 21 ----
PART_D={}
if "D" in PARTS:
  for nm in PART_D_MODELS:
    w=W5[nm]; PL=w["probe_layer"]
    print("\n"+"="*78); print(f"PART D: {nm}"); print("="*78)
    t0=time.time()
    model,processor,source=load_rope(device=DEVICE,half=False,input_size=BASE_RES,**w["kwargs"])
    model.eval().float()
    rh,_,nh_a=attn_attrs(model,source); n_heads=int(nh_a) if nh_a else w["n_heads"]
    base,_=ssdc_only(model,processor,source,NullCtx(),N_IMAGES_SSDC)
    floor,_=ssdc_only(model,processor,source,RopeGlobal(model,identity_like),N_IMAGES_SSDC)
    gate(f"{nm}: baseline reproduces W5","partD",abs(base[PL]-w["baseline"])<ANCHOR_TOL,
         f"{base[PL]:.6f} vs {w['baseline']}")
    ret=lambda ss: (ss[PL]-floor[PL])/(base[PL]-floor[PL])
    rows=[]
    for k in D_KS:
        topk=w["stageA"][:k]
        r,_=ssdc_only(model,processor,source,
                      RopeComposite(model,source,[(range(N_BLOCKS),per_head(topk,identity_like,n_heads))]),
                      N_IMAGES_SSDC)
        rows.append({"k":k,"arm":"top","heads":topk,"overlap":k,"retention":ret(r)})
        print(f"  k={k} top     heads {topk} retention {ret(r):.4f}")
        for d in range(D_DRAWS):
            rng=np.random.default_rng(D_SEED+1000*k+d+sum(ord(c) for c in nm))
            hs=sorted(rng.choice(n_heads,size=k,replace=False).tolist())
            rr,_=ssdc_only(model,processor,source,
                           RopeComposite(model,source,[(range(N_BLOCKS),per_head(hs,identity_like,n_heads))]),
                           N_IMAGES_SSDC)
            ov=sorted(set(hs)&set(topk))
            rows.append({"k":k,"arm":f"random_{d}","heads":hs,"overlap":len(ov),"retention":ret(rr)})
            print(f"  k={k} random{d} heads {hs} overlap {len(ov)} retention {ret(rr):.4f}")
    PART_D[nm]={"rows":rows,"baseline":base[PL],"floor":floor[PL],"runtime_s":round(time.time()-t0,1)}
    save(f"partD_{nm}",PART_D[nm])
    del model
    if DEVICE=="cuda": torch.cuda.empty_cache()

# ---- cell 23 ----
import pandas as pd
pd.set_option("display.width",250); pd.set_option("display.float_format",lambda v:f"{v:,.4f}")

if PART_A:
    print("PART A: top-1 cost over Cref, in points\n")
    for nm,e in PART_A.items():
        d=e["draw"]
        print(f"--- {nm}   regen r1 = h{d['regen_rank1']} (A rank {d['regen_rank1_A']}), "
              f"r2 = h{d['regen_rank2']} (A rank {d['regen_rank2_A']})")
        print(f"    random heads {d['random_heads']} with (A rank, C rank) "
              f"{ {h:(v['A'],v['C']) for h,v in d['ranks'].items()} }"
              + ("   [pool exhausted]" if d["exhaustive"] else ""))
        rows=[]
        for res,blk in e["by_res"].items():
            c=blk["conditions"]
            for order in ACC_ORDERS:
                cref=c["Cref"]["acc"][order]
                rnd=[(cref-c[k]["acc"][order])*100 for k in c if k.startswith("random_h")]
                for k in c:
                    if k in ("baseline","floor","Cref"): continue
                    cost=(cref-c[k]["acc"][order])*100
                    rows.append({"res":int(res),"order":order,"condition":k,"cost_pts":cost,
                                 "vs_random_mean":cost/np.mean(rnd) if np.mean(rnd)>1e-9 else np.nan})
        t=pd.DataFrame(rows)
        print(t.to_string(index=False))
        for res in sorted(t.res.unique()):
            for order in ACC_ORDERS:
                s=t[(t.res==res)&(t.order==order)]
                rn=s[s.condition.str.startswith("random_h")]["cost_pts"]
                for lbl in ("regen_r1","regen_r2"):
                    row=s[s.condition.str.startswith(lbl)]
                    if row.empty or rn.empty: continue
                    print(f"    {res} {order}: {lbl} costs {row.cost_pts.iloc[0]:.2f} pts vs random "
                          f"{rn.mean():.2f} (range {rn.min():.2f} to {rn.max():.2f}) "
                          f"= {row.cost_pts.iloc[0]/rn.mean():.1f}x")
        print()

# ---- cell 24 ----
if PART_B:
    print("PART B: SSDC noise on the three checkpoints added in W5\n")
    rows=[]
    for nm,e in PART_B.items():
        for b,r in e["by_block"].items():
            rows.append({"model":nm,"block":int(b),"point":r["point"],"boot_sd":r["boot_sd"],
                         "seed_sd":r["seed_sd"],"combined_sd":r["combined_sd"]})
    nb=pd.DataFrame(rows); print(nb.to_string(index=False))
    print("\n  naver's E1.2 combined SD was 0.00834 and DINOv3's 0.00077, for comparison.")
    print("  Use the block-11 value to judge stage C rank gaps on these models.")
    if PART_A or True:
        print("\n  stage C rank-1 vs rank-2 gaps against the measured noise:")
        for nm,e in PART_B.items():
            b11=e["by_block"].get("11")
            if not b11: continue
            print(f"    {nm}: combined SD at block 11 = {b11['combined_sd']:.5f}")

if PART_C:
    print("\nPART C: probe noise, in accuracy points\n")
    rows=[]
    for nm,e in PART_C.items():
        for cond,bb in e["conditions"].items():
            for b,hh in bb.items():
                for h in ("exact","row","col"):
                    rows.append({"model":nm,"condition":cond,"block":int(b),"head":h,
                                 "acc_pct":hh[h]["acc"]*100,"boot_sd":hh[h]["boot_sd"]*100,
                                 "seed_sd":hh[h]["seed_sd"]*100,"combined_sd":hh[h]["combined_sd"]*100})
    pc=pd.DataFrame(rows)
    print(pc.groupby(["model","head"])[["boot_sd","seed_sd","combined_sd"]].mean().round(3).to_string())
    print("\n  W5 used a proxy SD of 0.39 points for both models. Compare against the measured value.")

if PART_D:
    print("\nPART D: stage B, top-k against several random draws\n")
    for nm,e in PART_D.items():
        t=pd.DataFrame(e["rows"])
        t["loss"]=1-t["retention"]
        print(f"--- {nm}")
        for k in sorted(t.k.unique()):
            s=t[t.k==k]; top=s[s.arm=="top"].iloc[0]; rnd=s[s.arm!="top"]
            ratios=[(1-top.retention)/(1-r) if (1-r)>1e-9 else np.nan for r in rnd.retention]
            print(f"  k={k}: top loss {1-top.retention:.4f} | random losses "
                  f"{[f'{1-r:.4f}' for r in rnd.retention]} | ratios "
                  f"{[f'{x:.1f}x' for x in ratios]}  overlaps {rnd.overlap.tolist()}")
        print()

# ---- cell 26 ----
gdf=pd.DataFrame(GATES)
print("GATE REPORT\n"); print(gdf.to_string(index=False) if len(gdf) else "(none)")
n_fail=int((gdf.status=="FAIL").sum()) if len(gdf) else 0
# counted from the gate calls: 1 metric + 5 per layout x 2 + 1 composite = 12 pre-model,
# then 1 draw gate per part-A model, 5 setup + 6 per-resolution gates per part-A model
# (3 of the 6 only fire when WITH_SSDC_A), 2 per part-B model, 1 per part-C and part-D model.
per_a = 5 + (6 if WITH_SSDC_A else 2)
expected = 12 \
    + (len(PART_A_MODELS)*(1+per_a) if "A" in PARTS else 0) \
    + (len(PART_B_MODELS)*2 if "B" in PARTS else 0) \
    + (len(PART_C_MODELS)   if "C" in PARTS else 0) \
    + (len(PART_D_MODELS)   if "D" in PARTS else 0)
acct=len(GATES)>=expected
print(f"\ngates run {len(GATES)} | expected at least {expected} | accounting {'OK' if acct else 'SHORT'}")
print("="*78)
if n_fail==0 and acct: print("ALL GATES PASSED")
else:
    if n_fail: print(f"{n_fail} GATE(S) FAILED")
    if not acct: print("GATE ACCOUNTING SHORT: some gates did not run")
    for _,r in gdf[gdf.status=="FAIL"].iterrows():
        print(f"   - [{r['scope']}] {r['gate']}: {r['detail']}")
print("="*78)

# ---- cell 27 ----
export={
 "notebook":"W6","generated_utc":RUN_STAMP,"parts_run":PARTS,
 "environment":{"torch":torch.__version__,"timm":timm.__version__,"device":DEVICE,
                "dtype":"float32","repo_path":str(repo),
                "gpu":torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"},
 "measurement":{"n_images_acc":N_IMAGES_ACC,"n_images_ssdc":N_IMAGES_SSDC,
                "batch_size":BATCH_SIZE,"metric":METRIC,"acc_orders":ACC_ORDERS,
                "torch_seed":TORCH_SEED,"random_head_seed":RANDOM_HEAD_SEED,
                "base_res":BASE_RES,"extrap_res":EXTRAP_RES,"num_workers":0,
                "precision":"fp32; baseline and floor re-measured in-run per model per resolution"},
 "design_notes":{
   "part_A":"random heads drawn from outside the stage C top-D_TOP with a per-model seed; "
            "regenerator ranks 1 and 2 knocked out separately because the gap between them is "
            "0.7 SD on naver and has no measured noise scale on the other models",
   "part_B":"image term from cached per-image similarity matrices, gated against the repo's own "
            "SSDC; seed term from four RPI permutation seeds; blocks are the probe layer and 11",
   "anchors":"checked at 1000 images against W5 before the 5000-image sweep, because a different "
             "sample size gives a different point estimate",
   "excluded":"rope_ape from part A (head knockouts cost ~0 top-1 in W5, APE carries position at "
              "the probe layer); DINOv3 from part A (num_classes = 0)"},
 "w5_inputs":{k:{kk:vv for kk,vv in v.items() if kk!="kwargs"} for k,v in W5.items()},
 "draws":DRAWS,"part_A":PART_A,"part_B":PART_B,"part_C":PART_C,"part_D":PART_D,
 "gates":GATES,
 "gate_summary":{"total":len(GATES),"pass":int(sum(g["status"]=="PASS" for g in GATES)),
                 "fail":n_fail,"expected_at_least":expected,"accounting_ok":bool(acct)},
}
out=OUT_DIR/f"w6_all_{RUN_STAMP.replace(':','-')}.json"
out.write_text(json.dumps(export,indent=2))

rows=[]
for nm,e in PART_A.items():
    for res,blk in e["by_res"].items():
        for c,r in blk["conditions"].items():
            row={"model":nm,"res":int(res),"condition":c}
            if "ssdc" in r: row.update({"ssdc_probe":r["ssdc"][e["probe_layer"]],"ssdc_b11":r["ssdc"][11]})
            row.update({f"top1_{k}":v for k,v in r["acc"].items()})
            rows.append(row)
if rows: pd.DataFrame(rows).to_csv(OUT_DIR/"w6_partA.csv",index=False)
gdf.to_csv(OUT_DIR/"w6_gate_report.csv",index=False)

import shutil
archive=shutil.make_archive("w6_output","zip",OUT_DIR)
print("written:")
for p in sorted(OUT_DIR.iterdir()): print(f"  {p}  ({p.stat().st_size:,} bytes)")
print(f"\narchive: {archive} ({os.path.getsize(archive):,} bytes)")
try:
    from google.colab import files as cf; cf.download(archive)
except Exception: print("(not Colab: copy the archive back manually)")