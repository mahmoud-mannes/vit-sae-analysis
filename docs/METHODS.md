# Methods

This document gives the precise definitions and the implementation details for
the metrics and interventions used in the repository.

## Residual stream extraction

For each of the 12 blocks we capture the tensor that enters the block's attention
submodule. For the timm model that is `norm1(x)`, the input to `block.attn`. For
the transformers model that is `layernorm_before(hidden_states)`, the input to
`layer.attention`. Both are the post norm residual entering the block, so the two
models are measured on the same quantity. Capture is done with forward hooks in
`metrics/ssdc.py`, one per block. The hooks fold each batch into a running sum of
per image cosine similarity, so memory stays at one `T` by `T` matrix per block
regardless of how many images are streamed.

## SSDC (Spatial Similarity Distance Correlation)

Let `S` be the `T` by `T` cosine similarity matrix over patch tokens, averaged
over the image batch. Let `p_i` be the grid coordinate of token `i`, laid out in
row major order so token `k` sits at `(k // G, k % G)` for a `G` by `G` grid.
Define `D_ij = || p_i - p_j ||_1`. Then

```
SSDC = spearman( { S_ij }_{i<j}, { -D_ij }_{i<j} )
```

taken over the strict upper triangle of unordered token pairs. Spearman rank
correlation keeps the metric agnostic to the exact functional form relating
distance and similarity. The class token is dropped before the metric so the
token set is a clean square. For ViT-Base/16 at 224 that is 196 patch tokens on a
14 by 14 grid. `spatial_similarity_distance_correlation` in `metrics/ssdc.py`
implements this and is unit tested to return `+1` when similarity equals negative
distance, `-1` when it equals distance, and near zero for random inputs.

SSDC is a comparative probe, not an absolute measurement of a mechanism. Its
value across depth and across interventions is the informative signal.

## RPI (Random Permutation at Inference)

RPI shuffles the patch tokens while pinning the positional signal to the sequence
index. The hook in `main/model.py` acts on the patch embedding convolution output
of shape `[B, C, H, W]`. It permutes the `H*W` spatial positions with one random
permutation, so patch content moves but the coordinate each sequence slot will be
assigned stays fixed. Downstream the fixed positional signal attaches to the
fixed index. For APE that is the learned embedding added in index order. For RoPE
that is the rotation applied by sequence position inside attention. SSDC under RPI
uses the fixed grid coordinates, so it asks whether representational similarity
still tracks position once content has been scrambled. Structure that survives is
anchored to index rather than content.

## Component ablation

`interventions/ablation.py` provides `AblationController`, a context manager that
zero ablates a component at a set of layers. A ViT block computes

```
x = x + attn(norm1(x))     # attention sublayer
x = x + mlp(norm2(x))      # feedforward sublayer
```

Zero ablating a sublayer forces its output to zero, so the residual passes
through untouched. The mechanics differ by model because the two libraries lay
out the block differently.

- timm: `block.attn` and `block.mlp` each return a single tensor added to the
  residual. The hook returns `zeros_like(output)`.
- transformers: `layer.attention` returns a tuple whose first element is the
  context tensor, so the hook zeros that element. The MLP residual add lives
  inside `layer.output` (`ViTOutput.forward(hidden_states, input_tensor)` returns
  `dense(hidden_states) + input_tensor`), so ablating the MLP means returning
  `input_tensor`, which keeps the residual and drops the feedforward update.

Two modes are supported. `zero` ablates the listed layers. `keep_only` ablates
every layer except the listed ones, which keeps the component alive in a single
window. The controller only intercepts forward outputs, so it changes no weights
and is fully reversible on exit. The unit test confirms that ablating both attn
and mlp at every layer turns the stack into an exact identity for both layouts,
which verifies that the hooks target the right tensors.

Ablation composes with SSDC and RPI. The ablation hooks sit on the block
submodules, the SSDC capture hooks read the block inputs, and the RPI hook sits
on the patch embedding, so all three can be active at once. Because a component
at block `i` is ablated before its output reaches block `i+1`, the effect on a
per layer curve appears at block `i+1` and beyond.

## Fragility

`fragility = 1 - shifted_accuracy / baseline_accuracy`. Accuracy is top 1 over the
streamed sample, read straight from the ImageNet-1k head. The shift used in the
reference run is ImageNet-C Gaussian blur at severity 5, which maps to a
Gaussian sigma of 6.

## Effective rank

Given the singular values `s_1..s_r` of the token matrix, set `p_i = s_i / sum_j
s_j` and report `exp(-sum_i p_i log p_i)`. This is the effective rank of Roy and
Vetterli. It equals 1 for a rank one matrix (full collapse) and grows toward the
embedding dimension as energy spreads across directions. The implementation in
`metrics/effective_rank.py` computes it per image with a batched SVD and averages,
and offers a `center` option that subtracts the mean token first to isolate the
collapse toward a common vector that Dong et al. describe.

## Corruptions

`interventions/corruptions.py` reimplements the few ImageNet-C corruptions used
here in pure numpy and Pillow, so the notebook runs on a clean Colab without
ImageMagick or wand. The Gaussian blur matches the ImageNet-C math, so fragility
numbers line up with the reference run. `prep_data.py` prefers the full ImageNet-C
suite in `make_imagenet_c.py` when its heavy dependencies are importable, and
falls back to this light module otherwise.
