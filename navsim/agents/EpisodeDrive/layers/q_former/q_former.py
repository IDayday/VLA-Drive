import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v, need_weights=False)
        return out

class VisionQFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        # 1️⃣ Query Self-Attention
        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)

        # 2️⃣ Query → Vision Cross-Attention
        self.ln_cross = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)

        # 3️⃣ Feed Forward
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, queries, vision_tokens):
        # Self-Attention
        q = self.ln_self(queries)
        queries = queries + self.self_attn(q, q, q)

        # Cross-Attention (Q -> Vision)
        q = self.ln_cross(queries)
        queries = queries + self.cross_attn(q, vision_tokens, vision_tokens)

        # Feed Forward
        q = self.ln_ffn(queries)
        queries = queries + self.ffn(q)

        return queries


class VisionOnlyQFormer(nn.Module):
    def __init__(
        self,
        vision_dim=1536,
        hidden_dim=256,
        num_queries=64,
        num_layers=4,
        num_heads=8,
    ):
        super().__init__()

        # Vision projection
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Q-Former Blocks
        self.blocks = nn.ModuleList([
            VisionQFormerBlock(
                dim=hidden_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)

        # Init (important)
        # nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, scene_queries, vision_tokens):
        """
        vision_tokens: [B, 2800, 1536]
        return:        [B, 64, 256]
        """
        B = vision_tokens.size(0)

        # Project vision tokens
        vision_tokens = self.vision_proj(vision_tokens)

        # Expand queries
        queries = scene_queries.expand(B, -1, -1)

        # Q-Former layers
        for blk in self.blocks:
            queries = blk(queries, vision_tokens)

        return self.ln_out(queries)
