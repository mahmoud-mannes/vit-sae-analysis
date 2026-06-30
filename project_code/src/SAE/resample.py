@torch.no_grad()
def resample_dead_features(sae, optimizer, dead_indices, high_loss_examples, device):
    """
    Resamples dead SAE features following the Towards Monosemanticity approach.

    sae: SAE_Module instance
    optimizer: the AdamW optimizer training the SAE
    dead_indices: 1D LongTensor of feature indices (into hidden dim) that are dead
    high_loss_examples: tensor [n_dead, d_model] of input vectors with high
        reconstruction loss, one per dead feature, used to reinitialize them
    """
    if len(dead_indices) == 0:
        return

    n_dead = len(dead_indices)

    # --- Step 1: compute target norm to match existing live decoder columns ---
    # linear_down.weight has shape [d_model, d_hidden]; each column j is the
    # decoder direction for feature j.
    decoder_weight = sae.linear_down.weight  # [d_model, d_hidden]
    live_mask = torch.ones(decoder_weight.shape[1], dtype=torch.bool, device=device)
    live_mask[dead_indices] = False
    if live_mask.any():
        avg_live_norm = decoder_weight[:, live_mask].norm(dim=0).mean()
    else:
        avg_live_norm = torch.tensor(1.0, device=device)

    # --- Step 2: build new directions from high-loss examples ---
    # Normalize each example to unit norm, then scale to match live decoder norms
    # (scaled down a bit so the new feature doesn't immediately dominate).
    new_directions = high_loss_examples / (
        high_loss_examples.norm(dim=-1, keepdim=True) + 1e-8
    )
    scale_factor = 0.2  # conservative initial scale, per Anthropic's approach
    new_decoder_vecs = new_directions * avg_live_norm * scale_factor  # [n_dead, d_model]

    # --- Step 3: write new decoder weights (linear_down.weight columns) ---
    decoder_weight[:, dead_indices] = new_decoder_vecs.T

    # --- Step 4: write matching encoder weights/bias (linear_up[0]) ---
    encoder_linear = sae.linear_up[0]  # nn.Linear(d_model, d_hidden)
    # encoder_linear.weight has shape [d_hidden, d_model]; row j is feature j's
    # encoder direction. Point it the same way as the new decoder direction.
    encoder_linear.weight[dead_indices, :] = new_directions  # unit norm, encoder side
    encoder_linear.bias[dead_indices] = 0.0  # neutral bias so it can fire freely

    # --- Step 5: reset optimizer (Adam) moment buffers for resampled params ---
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is decoder_weight:
                state = optimizer.state.get(p, None)
                if state:
                    state["exp_avg"][:, dead_indices] = 0.0
                    state["exp_avg_sq"][:, dead_indices] = 0.0
            elif p is encoder_linear.weight:
                state = optimizer.state.get(p, None)
                if state:
                    state["exp_avg"][dead_indices, :] = 0.0
                    state["exp_avg_sq"][dead_indices, :] = 0.0
            elif p is encoder_linear.bias:
                state = optimizer.state.get(p, None)
                if state:
                    state["exp_avg"][dead_indices] = 0.0
                    state["exp_avg_sq"][dead_indices] = 0.0


def get_high_loss_examples(x, x_reconstructed, n_needed):
    """Returns the n_needed input vectors with highest per-token reconstruction loss."""
    per_token_loss = ((x - x_reconstructed) ** 2).sum(dim=-1)  # [batch]
    n_needed = min(n_needed, x.shape[0])
    top_idx = torch.topk(per_token_loss, n_needed).indices
    return x[top_idx]