import math
import torch
from torch import nn


def grid_positions(height, width):
    x, y = torch.meshgrid((torch.arange(height)+.5)/height,
                          (torch.arange(width)+.5)/width, indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2)


class QueryBank(nn.Module):
    def __init__(self, hidden_size, grid_hw, seed):
        super().__init__()
        # Initialization cannot perturb baseline or proposal RNG, including CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.queries = nn.Parameter(torch.randn(math.prod(grid_hw), hidden_size)*.02)
            self.position = nn.Sequential(nn.Linear(2, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.register_buffer("xy", grid_positions(*grid_hw), persistent=False)

    def forward(self):
        return self.queries + self.position(self.xy)


def select_tail_lora(language_model, rank=32, last_layers=8, seed=4201):
    """Attach PEFT only to actual tail attention projections; return exact names."""
    from peft import LoraConfig, get_peft_model
    layers = language_model.layers
    chosen = list(range(max(0, len(layers)-last_layers), len(layers)))
    names = [f"layers.{i}.self_attn.{p}" for i in chosen for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    actual = dict(language_model.named_modules())
    if not all(n in actual for n in names):
        raise ValueError("LoRA selection does not match actual Qwen3 modules")
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
        torch.manual_seed(seed)
        wrapped = get_peft_model(language_model, LoraConfig(r=rank, lora_alpha=rank,
                    target_modules=names, lora_dropout=0., bias="none"))
    return wrapped, names
