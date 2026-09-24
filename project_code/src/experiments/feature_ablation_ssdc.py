"""SAE feature ablation: impact on SSDC.

Question
--------
If we remove the SAE features that carry position (row / column) information
from one site of the network, how much of the layer-wise spatial structure
(SSDC, measured under RPI) do we lose, compared with

    no_SAE_baseline           the untouched model
    SAE_baseline              the SAE reconstruction with nothing ablated
                              (this is the right reference for the ablations,
                              because the reconstruction itself perturbs the model)
    SAE_{k}_features_ablated  the k top positional features, ablated
    {k}_random_features_ablated  k random features, ablated (control)

The top positional features come from the selectivity JSON written by the SAE
feature analysis (top_selective_features_per_position_{block}{seed}.json). Only
the order in that file is used: the first k unique features are ablated, exactly
as SAE_causal.feature_ablation.load_top_features returns them.

Where the ablation happens
--------------------------
    block="residual"   residual stream entering block `layer` (before its forward pass)
    block="attention"  output of the attention sublayer of block `layer`
    block="mlp"        output of the MLP sublayer of block `layer`
The SAE for that site must exist, see "SAE checkpoints" below.

SAE checkpoints
---------------
`sae_dir` is the folder that holds the checkpoints. It is a parameter (or the
SAE_MODELS_DIR environment variable). The file inside it is looked up as

    SAE_{block}_{model_type}_{layer}_TOP{sae_topk}{suffix}.pt

e.g. SAE_residual_APE_2_TOP64.pt, SAE_residual_APE_2_TOP64_second_seed.pt.
Pass `sae_filename` to use another template ({block} {model_type} {layer}
{sae_topk} {suffix} {seed} are available).

Selectivity features
--------------------
`selectivity_scores_path` defaults to

    <repo>/results/runs/SAE_20k_images_analysis/<original_results|second_seed|...>/
        top_selective_features_per_position_{block}{suffix}.json

built from this file's location with pathlib, so it works on any machine.

"seed" is the repository's internal model label, not a random seed:
    APE : 1 google/vit-base-patch16-224, 2 DINOv1, 3 AugReg, 4 SAM
    RoPE: 1 naver ViT-B RoPE, 2 DINOv3 (loaded at 224)
Random choices in this script (random features, RPI permutations) are seeded
with `condition_seed` / `rpi_seed` and are identical across conditions, so
conditions are compared on the same images and the same permutations.

Run
---
    python feature_ablation_ssdc.py --model-type APE --seed 1 --sae-dir /path/to/SAE_Models --plot
    python feature_ablation_ssdc.py --model-type RoPE --sae-dir ... --k-values 8 16 35

From a notebook (repo `project_code/src` on sys.path):
    from experiments.feature_ablation_ssdc import run_all
    out = run_all(model_type="APE", seed=1, sae_dir="/content/drive/MyDrive/SAE_Models", do_plot=True)

The images are decoded once and cached, so adding conditions is cheap.
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

# Make this file work as a script, as `python -m`, and as an import.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import common
from interventions.ablation import num_blocks
from main import load_models
from main.load_models import get_patch_embed_conv, num_prefix_tokens
from metrics.ssdc import SSDCAccumulator, _register_ssdc_accumulator_hooks
from SAE_causal.feature_ablation import ablate_features, load_top_features


# --------------------------------------------------------------------------- #
# Constants: models, paths, naming conventions
# --------------------------------------------------------------------------- #

BLOCKS = ("residual", "attention", "mlp")
AXES = ("both", "row", "column")

# Where the SAE feature analysis results live (selectivity JSONs).
ANALYSIS_DIR = Path(common.RESULTS_DIR) / "runs" / "SAE_20k_images_analysis"

# Internal model label -> file-name suffix / results sub-folder.
SEED_SUFFIX = {1: "", 2: "_second_seed", 3: "_third_seed", 4: "_fourth_seed"}
SEED_RESULTS_SUBDIR = {
    1: "original_results",
    2: "second_seed",
    3: "third_seed",
    4: "fourth_seed",
}

# Layer the SAEs / feature ablations were done at, per model family.
DEFAULT_LAYER = {"APE": 2, "RoPE": 4}

DEFAULT_SAE_FILENAME = "SAE_{block}_{model_type}_{layer}_TOP{sae_topk}{suffix}.pt"
SAE_DIR_ENV_VAR = "SAE_MODELS_DIR"


@dataclass(frozen=True)
class ModelSpec:
    model_type: str      # "APE" | "RoPE"
    seed: int            # internal model label (1..4)
    loader: str          # function name in main.load_models
    loader_kwargs: dict  # forwarded to that loader
    checkpoint: str      # human readable name, stored in the results


# RoPE models are loaded at 224 like every reference run (see
# experiments/rope/README.md: load_rope without input_size gives DINOv3 a 16x16 grid).
MODEL_SPECS = {
    ("APE", 1): ModelSpec("APE", 1, "load_ape", {}, "google/vit-base-patch16-224"),
    ("APE", 2): ModelSpec(
        "APE", 2, "load_ape_timm",
        {"model_name": "vit_base_patch16_224.dino"}, "vit_base_patch16_224.dino",
    ),
    ("APE", 3): ModelSpec(
        "APE", 3, "load_ape_timm",
        {"model_name": "vit_base_patch16_224.augreg_in1k"}, "vit_base_patch16_224.augreg_in1k",
    ),
    ("APE", 4): ModelSpec(
        "APE", 4, "load_ape_timm",
        {"model_name": "vit_base_patch16_224.sam_in1k"}, "vit_base_patch16_224.sam_in1k",
    ),
    ("RoPE", 1): ModelSpec(
        "RoPE", 1, "load_rope",
        {"input_size": 224}, "vit_base_patch16_rope_224.naver_in1k",
    ),
    ("RoPE", 2): ModelSpec(
        "RoPE", 2, "load_rope",
        {"model_name": "vit_base_patch16_dinov3.lvd1689m", "input_size": 224},
        "vit_base_patch16_dinov3.lvd1689m",
    ),
}


def normalize_model_type(model_type):
    """'ape' / 'APE' -> 'APE', 'rope' / 'RoPE' -> 'RoPE' (the spelling used in the JSON keys)."""
    key = str(model_type).strip().lower()
    if key == "ape":
        return "APE"
    if key == "rope":
        return "RoPE"
    raise ValueError(f"model_type must be 'APE' or 'RoPE', got {model_type!r}")


def get_model_spec(model_type, seed):
    key = (normalize_model_type(model_type), int(seed))
    if key not in MODEL_SPECS:
        raise ValueError(f"no model registered for {key}; available: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[key]


def load_model_from_spec(spec, device=None):
    """Load the model in float32 (the SAEs are float32, so no half precision here)."""
    loader = getattr(load_models, spec.loader)
    return loader(device=device, half=False, **spec.loader_kwargs)


def resolve_sae_dir(sae_dir=None):
    sae_dir = sae_dir or os.environ.get(SAE_DIR_ENV_VAR)
    if not sae_dir:
        raise ValueError(
            f"Tell me where the SAE checkpoints are: pass sae_dir=... (--sae-dir) "
            f"or set the {SAE_DIR_ENV_VAR} environment variable."
        )
    return Path(sae_dir).expanduser()


def sae_checkpoint_path(sae_dir, block, model_type, layer, sae_topk, seed, filename_template=None):
    template = filename_template or DEFAULT_SAE_FILENAME
    name = template.format(
        block=block,
        model_type=model_type,
        layer=layer,
        sae_topk=sae_topk,
        suffix=SEED_SUFFIX[int(seed)],
        seed=int(seed),
    )
    return resolve_sae_dir(sae_dir) / name


def default_selectivity_path(block, seed):
    """<repo>/results/runs/SAE_20k_images_analysis/<subdir>/top_selective_features_per_position_{block}{suffix}.json"""
    seed = int(seed)
    return (
        ANALYSIS_DIR
        / SEED_RESULTS_SUBDIR[seed]
        / f"top_selective_features_per_position_{block}{SEED_SUFFIX[seed]}.json"
    )


def results_path(prefix, model_type, seed, block, layer, output_dir=None, ext=".json", tag=""):
    """<output_dir or results/>/<prefix>_<model>_seed<seed>_<block>_layer<layer>[_<tag>].json"""
    directory = Path(output_dir) if output_dir else Path(common.RESULTS_DIR)
    suffix = f"_{tag}" if tag else ""
    return directory / f"{prefix}_{model_type.lower()}_seed{seed}_{block}_layer{layer}{suffix}{ext}"


def write_results(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    common.save_json(obj, str(path))
    print(f"saved {path}")
    return path


# --------------------------------------------------------------------------- #
# SAE and feature selection
# --------------------------------------------------------------------------- #

def load_sae(
    checkpoint_path,
    d_model=768,
    d_multiplier=5,
    architecture="topk",
    sae_topk=64,
    device=None,
):
    """Build the SAE and load its weights (frozen, eval mode).

    The decoder bias / mean is part of the state dict, so nothing has to be
    estimated from activations first.
    """
    from SAE import SAE as SparseAutoencoder  # local import: pulls in the whole SAE package

    sae = SparseAutoencoder(
        d_model=d_model,
        d_hidden=d_model * d_multiplier,
        architecture=architecture,
        k=sae_topk,
    )
    sae.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    sae = sae.to(device or load_models.get_device()).eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)

    if architecture == "batchtopk":
        # The hook feeds [B, T, d_hidden] latents, but the training-time batchtopk
        # rule assumes [tokens, d_hidden]. Deploy it as the JumpReLU it is converted to.
        if float(sae.inference_threshold.max()) <= 0:
            raise ValueError(
                "batchtopk checkpoint has no estimated inference threshold; "
                "run SAE.estimate_threshold before saving it."
            )
        sae.use_inference_threshold = True
    return sae


def get_positional_features(selectivity_scores_path, model_type, layer, axis="both"):
    """Ordered unique positional features for this model/layer (see load_top_features)."""
    path = Path(selectivity_scores_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"selectivity file not found: {path}\n"
            f"Pass selectivity_scores_path=... (--selectivity-path), or skip the top-feature "
            f"conditions with k_values=[]."
        )
    try:
        features = load_top_features(str(path), model_type, layer, axis=axis)
    except KeyError:
        with open(path) as f:
            available = sorted(json.load(f))
        raise KeyError(
            f"'{model_type}_layer_{layer}' is not in {path.name}; it has {available}"
        ) from None
    return [int(f) for f in features]


def default_k_values(n_positional, n_levels=5):
    """Five evenly spaced doses ending at all positional features (20 -> 4, 8, 12, 16, 20)."""
    if n_positional < 1:
        return []
    return sorted({max(1, math.ceil(n_positional * i / n_levels)) for i in range(1, n_levels + 1)})


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #

@dataclass
class Condition:
    """One experimental condition.

    kind: "no_sae" (untouched model), "sae_baseline" (reconstruction only),
          "top" (top positional features ablated), "random" (random features ablated).
    features: SAE feature indices to zero (None for "no_sae", [] for "sae_baseline").
    """

    name: str
    kind: str
    k: int = 0
    draw: Optional[int] = None
    features: Optional[List[int]] = None

    def spec(self):
        return {"kind": self.kind, "k": self.k, "draw": self.draw, "features": self.features}


def build_conditions(
    d_hidden,
    positional_features,
    k_values,
    random_k_values=None,
    random_draws=1,
    seed=0,
    exclude_positional_from_random=False,
    axis="both",
):
    """Baselines, then top-k ablations, then matched random-k controls.

    random_k_values defaults to k_values. random_draws=0 drops the random control.
    Random features are drawn from a generator seeded with (seed, k, draw).
    """
    k_values = [int(k) for k in (k_values or [])]
    random_k_values = k_values if random_k_values is None else [int(k) for k in random_k_values]
    positional_features = [int(f) for f in (positional_features or [])]

    conditions = [
        Condition("no_SAE_baseline", "no_sae"),
        Condition("SAE_baseline", "sae_baseline", features=[]),
    ]

    axis_tag = "" if axis == "both" else f"{axis}_"
    for k in k_values:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > len(positional_features):
            raise ValueError(
                f"k={k} but only {len(positional_features)} positional features are available "
                f"(axis={axis!r}). Use smaller k_values."
            )
        conditions.append(
            Condition(
                f"SAE_{k}_{axis_tag}features_ablated", "top", k=k,
                features=positional_features[:k],
            )
        )

    pool = np.arange(int(d_hidden))
    if exclude_positional_from_random:
        pool = np.setdiff1d(pool, positional_features)
    for k in random_k_values:
        if k < 1 or k > len(pool):
            raise ValueError(f"random k={k} must be between 1 and {len(pool)}")
        for draw in range(int(random_draws)):
            rng = np.random.default_rng([int(seed), k, draw])
            features = sorted(int(f) for f in rng.choice(pool, size=k, replace=False))
            name = f"{k}_random_features_ablated"
            if random_draws > 1:
                name += f"_draw{draw}"
            conditions.append(Condition(name, "random", k=k, draw=draw, features=features))

    names = [c.name for c in conditions]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate condition names: {names}")
    return conditions


@contextmanager
def ablation_context(model, source, model_type, sae, layer, block, condition):
    """Attach the feature-ablation hook for a condition; always removes it on exit.

    "no_sae" attaches nothing. "sae_baseline" attaches the hook with no features,
    i.e. plain SAE reconstruction.
    """
    if condition.kind == "no_sae":
        yield
        return
    handle = ablate_features(
        model=model,
        source=source,
        model_type=model_type,
        SAE=sae,
        layer=layer,
        block=block,
        features_to_remove=list(condition.features),
        top_features=False,
        random_features=False,
        selectivity_scores_path=None,
        k=0,
    )
    try:
        yield
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
# Running the model on cached batches
# --------------------------------------------------------------------------- #

def install_seeded_rpi_hook(model, source, seed):
    """Random Permutation at Inference, like main.model.predict, but seeded.

    Every forward pass (one batch) draws a fresh permutation of the patch tokens
    right after the patch embedding. Because the generator is created from `seed`
    here, every condition (and every layer probed) sees the same sequence of
    permutations as long as it runs the same batches.
    """
    generator = torch.Generator().manual_seed(int(seed))

    def hook(module, inputs, output):
        batch, channels, height, width = output.shape
        flat = output.reshape(batch, channels, -1)
        permutation = torch.randperm(height * width, generator=generator).to(output.device)
        return flat[:, :, permutation].reshape(batch, channels, height, width)

    return get_patch_embed_conv(model, source).register_forward_hook(hook)


def run_forward(model, source, batches):
    """One forward pass over the cached batches (hooks do the work)."""
    with torch.no_grad():
        for pixel_values, _labels in batches:
            pixel_values = common.prepare_pixel_values(pixel_values, model, half=False)
            common.forward_logits(model, source, pixel_values)


def evaluate_ssdc_on_batches(model, source, batches, RPI=True, rpi_seed=0, metric="manhattan"):
    """Per-layer SSDC (same quantity as metrics.ssdc.evaluate_ssdc) on cached batches."""
    accumulator = SSDCAccumulator(n_prefix=num_prefix_tokens(model, source), spatial_metric=metric)
    handles = _register_ssdc_accumulator_hooks(model, source, accumulator)
    if RPI:
        handles.append(install_seeded_rpi_hook(model, source, rpi_seed))
    try:
        run_forward(model, source, batches)
    finally:
        for handle in handles:
            handle.remove()
    return [float(accumulator.ssdc(i)) for i in sorted(accumulator.keys())]


def run_condition(
    model, source, model_type, batches, sae, layer, block, condition,
    RPI=True, rpi_seed=0, metric="manhattan",
):
    """Run one condition and return its per-layer SSDC curve.

    The model is only modified inside the ablation context and is restored afterwards.
    """
    with ablation_context(model, source, model_type, sae, layer, block, condition):
        return evaluate_ssdc_on_batches(
            model, source, batches, RPI=RPI, rpi_seed=rpi_seed, metric=metric
        )


# --------------------------------------------------------------------------- #
# Setup shared by feature_ablation_ssdc.py and feature_ablation_probe.py
# --------------------------------------------------------------------------- #

@dataclass
class Experiment:
    model_type: str
    seed: int
    spec: ModelSpec
    layer: int
    block: str
    model: Any
    processor: Any
    source: str
    n_layers: int
    n_prefix: int
    sae: Any
    sae_path: Path
    architecture: str
    sae_topk: int
    selectivity_path: Optional[Path]
    axis: str
    positional_features: List[int]
    conditions: List[Condition]
    batches: list
    number_images: int
    batch_size: int
    condition_seed: int
    exclude_positional_from_random: bool

    def metadata(self):
        """JSON-safe description of the setup, stored in every results file."""
        return {
            "model_type": self.model_type,
            "seed": self.seed,
            "model_checkpoint": self.spec.checkpoint,
            "source": self.source,
            "layer": self.layer,
            "block": self.block,
            "n_layers": self.n_layers,
            "n_prefix_tokens": self.n_prefix,
            "number_images": self.number_images,
            "batch_size": self.batch_size,
            "sae": {
                "path": str(self.sae_path),
                "architecture": self.architecture,
                "sae_topk": self.sae_topk,
                "d_model": int(self.sae.d_model),
                "d_hidden": int(self.sae.d_hidden),
            },
            "selectivity_scores_path": str(self.selectivity_path) if self.selectivity_path else None,
            "axis": self.axis,
            "n_positional_features": len(self.positional_features),
            "positional_features": self.positional_features,
            "condition_seed": self.condition_seed,
            "exclude_positional_from_random": self.exclude_positional_from_random,
        }


def get_dataset(dataset=None, hf_token=None, shuffle=False, data_seed=0):
    """Use the dataset you pass in, otherwise stream ImageNet-1k validation."""
    if dataset is not None:
        return dataset
    return common.load_imagenet(token=hf_token, shuffle=shuffle, seed=data_seed)


def setup_experiment(
    *,
    model_type,
    seed,
    layer,
    block,
    sae_dir,
    sae_filename,
    sae_topk,
    d_model,
    d_multiplier,
    architecture,
    selectivity_scores_path,
    axis,
    k_values,
    random_k_values,
    random_draws,
    exclude_positional_from_random,
    condition_seed,
    dataset,
    hf_token,
    shuffle_dataset,
    data_seed,
    number_images,
    batch_size,
):
    """Resolve paths, load model + SAE, build conditions, cache the image batches.

    Cheap checks (files exist, keys exist) run before anything is downloaded.
    """
    model_type = normalize_model_type(model_type)
    seed = int(seed)
    spec = get_model_spec(model_type, seed)
    layer = DEFAULT_LAYER[model_type] if layer is None else int(layer)
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}")

    # -- cheap checks first ---------------------------------------------------
    sae_path = sae_checkpoint_path(sae_dir, block, model_type, layer, sae_topk, seed, sae_filename)
    if not sae_path.is_file():
        raise FileNotFoundError(
            f"SAE checkpoint not found: {sae_path}\n"
            f"Check sae_dir (--sae-dir / {SAE_DIR_ENV_VAR}) or sae_filename."
        )

    wants_top = k_values is None or len(k_values) > 0
    selectivity_path, positional_features = None, []
    if wants_top or exclude_positional_from_random:
        selectivity_path = (
            Path(selectivity_scores_path) if selectivity_scores_path
            else default_selectivity_path(block, seed)
        )
        positional_features = get_positional_features(selectivity_path, model_type, layer, axis)
    if k_values is None:
        k_values = default_k_values(len(positional_features))

    # -- heavy loading --------------------------------------------------------
    model, processor, source = load_model_from_spec(spec)
    n_layers = num_blocks(model, source)
    layer = common.validate_layer_indices([layer], n_layers, name="layer")[0]

    sae = load_sae(
        sae_path, d_model=d_model, d_multiplier=d_multiplier,
        architecture=architecture, sae_topk=sae_topk, device=common.model_device(model),
    )

    conditions = build_conditions(
        d_hidden=sae.d_hidden,
        positional_features=positional_features,
        k_values=k_values,
        random_k_values=random_k_values,
        random_draws=random_draws,
        seed=condition_seed,
        exclude_positional_from_random=exclude_positional_from_random,
        axis=axis,
    )

    dataset = get_dataset(dataset, hf_token=hf_token, shuffle=shuffle_dataset, data_seed=data_seed)
    batches = common.collect_batches(dataset, processor, source, number_images, batch_size)
    if not batches:
        raise ValueError("no batches were collected from the dataset")

    return Experiment(
        model_type=model_type, seed=seed, spec=spec, layer=layer, block=block,
        model=model, processor=processor, source=source, n_layers=n_layers,
        n_prefix=num_prefix_tokens(model, source), sae=sae, sae_path=sae_path,
        architecture=architecture, sae_topk=sae_topk, selectivity_path=selectivity_path,
        axis=axis, positional_features=positional_features, conditions=conditions,
        batches=batches, number_images=int(sum(labels.numel() for _, labels in batches)),
        batch_size=int(batch_size), condition_seed=int(condition_seed),
        exclude_positional_from_random=bool(exclude_positional_from_random),
    )


# --------------------------------------------------------------------------- #
# Experiment orchestration
# --------------------------------------------------------------------------- #

def run_all(
    dataset=None,
    model_type="APE",
    seed=1,
    layer=None,
    block="residual",
    # SAE
    sae_dir=None,
    sae_filename=None,
    sae_topk=64,
    d_model=768,
    d_multiplier=5,
    architecture="topk",
    # features
    selectivity_scores_path=None,
    axis="both",
    k_values=None,
    random_k_values=None,
    random_draws=1,
    exclude_positional_from_random=False,
    condition_seed=0,
    # evaluation
    number_images=1000,
    batch_size=128,
    RPI=True,
    rpi_seed=0,
    metric="manhattan",
    shuffle_dataset=False,
    data_seed=0,
    hf_token=None,
    # outputs
    output_dir=None,
    tag="",
    save=True,
    do_plot=False,
):
    """Run every condition and return {metadata, conditions, curves, summary}.

    dataset               a streamed dataset of {'image', 'label'}; None streams ImageNet-1k validation.
    layer                 None -> 2 for APE, 4 for RoPE (the layers the SAEs were trained on).
    k_values              how many top positional features to ablate. None -> five doses ending at
                          all positional features; [] -> baselines (and random controls) only.
    random_k_values       k for the random controls. None -> same as k_values.
    random_draws          random draws per k (0 disables the random control).
    exclude_positional_from_random  draw random features only from non-positional ones.
    RPI / rpi_seed        SSDC under Random Permutation at Inference; the permutation sequence is
                          the same for every condition.
    output_dir            None -> <repo>/results/ (scratch, git-ignored); figures go to results/figures/.
    tag                   appended to the results file name (keeps runs with different settings apart).
    """
    exp = setup_experiment(
        model_type=model_type, seed=seed, layer=layer, block=block,
        sae_dir=sae_dir, sae_filename=sae_filename, sae_topk=sae_topk, d_model=d_model,
        d_multiplier=d_multiplier, architecture=architecture,
        selectivity_scores_path=selectivity_scores_path, axis=axis, k_values=k_values,
        random_k_values=random_k_values, random_draws=random_draws,
        exclude_positional_from_random=exclude_positional_from_random,
        condition_seed=condition_seed, dataset=dataset, hf_token=hf_token,
        shuffle_dataset=shuffle_dataset, data_seed=data_seed,
        number_images=number_images, batch_size=batch_size,
    )
    print(
        f"{exp.model_type} seed {exp.seed} | {exp.block} @ layer {exp.layer} | "
        f"{exp.number_images} images | {len(exp.positional_features)} positional features | "
        f"{len(exp.conditions)} conditions"
    )

    curves = {}
    for condition in exp.conditions:
        print(f"-- condition: {condition.name} --")
        start = time.time()
        curves[condition.name] = run_condition(
            exp.model, exp.source, exp.model_type, exp.batches, exp.sae,
            exp.layer, exp.block, condition, RPI=RPI, rpi_seed=rpi_seed, metric=metric,
        )
        print("   ", [round(x, 3) for x in curves[condition.name]], f"({time.time() - start:.1f}s)")

    out = {
        "experiment": "feature_ablation_ssdc",
        **exp.metadata(),
        "RPI": bool(RPI),
        "rpi_seed": int(rpi_seed),
        "metric": metric,
        "conditions": {c.name: c.spec() for c in exp.conditions},
        "curves": curves,
        "summary": {name: common.summarize_curve(curve) for name, curve in curves.items()},
    }
    print_summary(curves, exp.layer)

    if save:
        write_results(
            out,
            results_path("feature_ablation_ssdc", exp.model_type, exp.seed, exp.block, exp.layer, output_dir, tag=tag),
        )
    if do_plot:
        figure = results_path(
            "feature_ablation_ssdc", exp.model_type, exp.seed, exp.block, exp.layer,
            output_dir=common.FIGURES_DIR, ext=".png", tag=tag,
        )
        make_plots(out, save_path=figure)
    return out


def print_summary(curves, layer):
    print("\n==== SSDC per condition ====")
    header = f"{'condition':32s} {'@layer':>7s} {'peak':>6s} {'peak_L':>7s} {'final':>7s} {'auc':>7s}"
    print(header)
    print("-" * len(header))
    for name, curve in curves.items():
        s = common.summarize_curve(curve)
        at_layer = curve[layer] if layer < len(curve) else float("nan")
        print(
            f"{name:32s} {at_layer:7.3f} {s['peak']:6.3f} {s['peak_layer']:7d} "
            f"{s['final']:7.3f} {s['auc']:7.3f}"
        )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def make_plots(out, save_path=None):
    """Two panels sharing the baselines: top positional features vs random features."""
    import matplotlib.pyplot as plt

    curves, conditions = out["curves"], out["conditions"]

    def pick(*kinds):
        return {n: c for n, c in curves.items() if conditions[n]["kind"] in kinds}

    baselines = pick("no_sae", "sae_baseline")
    base_styles = {
        "no_SAE_baseline": dict(color="black", linewidth=2.0),
        "SAE_baseline": dict(color="black", linestyle="--", linewidth=1.5),
    }
    ylabel = "SSDC under RPI" if out.get("RPI", True) else "SSDC"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    panels = (("top", "top positional features ablated", "viridis"),
              ("random", "random features ablated", "autumn"))
    for ax, (kind, title, cmap_name) in zip(axes, panels):
        group = pick(kind)
        cmap = plt.get_cmap(cmap_name)
        styles = dict(base_styles)
        for i, name in enumerate(group):
            styles[name] = dict(color=cmap(0.85 * i / max(len(group) - 1, 1)))
        common.plot_curves(
            {**baselines, **group},
            title=f"{out['model_type']} seed {out['seed']} ({out['block']}, layer {out['layer']}): {title}",
            ylabel=ylabel, ax=ax, styles=styles,
        )
        ax.axvline(out["layer"], color="0.6", linestyle=":", linewidth=1)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        print(f"saved {save_path}")
    return fig


# --------------------------------------------------------------------------- #
# Command line (shared with feature_ablation_probe.py)
# --------------------------------------------------------------------------- #

def add_shared_cli_args(parser):
    parser.add_argument("--model-type", choices=["APE", "RoPE", "ape", "rope"], default="APE")
    parser.add_argument("--seed", type=int, default=1, choices=sorted(SEED_SUFFIX),
                        help="internal model label (not a random seed); see the module docstring")
    parser.add_argument("--layer", type=int, default=None, help="default: 2 for APE, 4 for RoPE")
    parser.add_argument("--block", choices=BLOCKS, default="residual")

    parser.add_argument("--sae-dir", default=None,
                        help=f"folder with the SAE checkpoints (or set ${SAE_DIR_ENV_VAR})")
    parser.add_argument("--sae-filename", default=None,
                        help=f"filename template, default {DEFAULT_SAE_FILENAME}")
    parser.add_argument("--sae-topk", type=int, default=64, help="the SAE's own TopK (the TOP{k} in the filename)")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-multiplier", type=int, default=5)
    parser.add_argument("--architecture", default="topk", choices=["topk", "batchtopk", "relu_l1", "jumprelu"])

    parser.add_argument("--selectivity-path", default=None,
                        help="top_selective_features_per_position JSON; default is built from the repo layout")
    parser.add_argument("--axis", choices=AXES, default="both")
    parser.add_argument("--k-values", type=int, nargs="*", default=None,
                        help="numbers of top positional features to ablate; default: five doses up to all of them")
    parser.add_argument("--random-k-values", type=int, nargs="*", default=None,
                        help="k for the random controls; default: same as --k-values")
    parser.add_argument("--random-draws", type=int, default=1, help="0 disables the random control")
    parser.add_argument("--exclude-positional-from-random", action="store_true")
    parser.add_argument("--condition-seed", type=int, default=0)

    parser.add_argument("--number-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-rpi", action="store_true", help="evaluate on unpermuted (clean) inputs")
    parser.add_argument("--rpi-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag", default="", help="appended to the results file name")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser


def shared_kwargs_from_args(args):
    """Turn the parsed shared arguments into run_all keyword arguments."""
    return dict(
        model_type=args.model_type, seed=args.seed, layer=args.layer, block=args.block,
        sae_dir=args.sae_dir, sae_filename=args.sae_filename, sae_topk=args.sae_topk,
        d_model=args.d_model, d_multiplier=args.d_multiplier, architecture=args.architecture,
        selectivity_scores_path=args.selectivity_path, axis=args.axis,
        k_values=args.k_values, random_k_values=args.random_k_values,
        random_draws=args.random_draws,
        exclude_positional_from_random=args.exclude_positional_from_random,
        condition_seed=args.condition_seed, number_images=args.number_images,
        batch_size=args.batch_size, RPI=not args.no_rpi, rpi_seed=args.rpi_seed,
        data_seed=args.data_seed, hf_token=args.hf_token, output_dir=args.output_dir,
        tag=args.tag, save=not args.no_save, do_plot=args.plot,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="SAE feature ablation: impact on SSDC.")
    add_shared_cli_args(parser)
    parser.add_argument("--metric", choices=["manhattan", "euclidean"], default="manhattan")
    parser.add_argument("--shuffle-dataset", action="store_true")
    args = parser.parse_args(argv)
    return run_all(
        **shared_kwargs_from_args(args), metric=args.metric, shuffle_dataset=args.shuffle_dataset
    )


if __name__ == "__main__":
    main()
