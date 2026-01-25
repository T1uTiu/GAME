import numpy as np
import torch
import torch.nn.functional as F
import torch.onnx.operators
from torch import nn


class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False, rotary_embed=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Q, K, V projections
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)

        # Final linear layer after concatenation
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Rotary Embeddings
        self.rotary_embed = rotary_embed

        # Initialization parameters
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if bias:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, C)
        # key_padding_mask: (B, L)
        batch_size, seq_len, embed_dim = x.size()

        # Project inputs to Q, K, V
        Q, K, V = torch.split(self.in_proj(x), self.embed_dim, dim=-1)

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)

        # Apply RoPE
        if self.rotary_embed is not None:
            Q = self.rotary_embed.rotate_queries_or_keys(Q)
            K = self.rotary_embed.rotate_queries_or_keys(K)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, L, L)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand mask to match attention scores shape
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 1, -np.inf)  # Masked positions are set to -inf

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to V
        attn_output = torch.matmul(attn_weights, V)  # (B, H, L, D)

        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (B, L, C)

        # Final linear projection
        output = self.out_proj(attn_output)  # (B, L, C)

        return output
