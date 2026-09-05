"""Content diagnostics; fixed slot offsets are explicitly subtracted."""
import torch


@torch.no_grad()
def semantic_content_diagnostics(tokens):
    values = tokens.detach().float()
    centered = values - values.mean(dim=1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probabilities = energy / energy.sum(-1, keepdim=True).clamp_min(1e-30)
    rank = (-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)).exp()
    result = {"semantic_slot_centered_rms": centered.square().mean().sqrt(),
              "semantic_effective_rank": rank.mean()}
    if values.shape[0] > 1:
        # A fixed learned query identity alone cancels in this statistic.
        content = values - values.mean(dim=0, keepdim=True)
        result["semantic_cross_scene_content_rms"] = content.square().mean().sqrt()
        slot_content = content - content.mean(dim=1, keepdim=True)
        result["semantic_cross_scene_slot_content_rms"] = slot_content.square().mean().sqrt()
    return result
