"""
author:liuyoude
date:2021-06-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleDotProductAttention(nn.Module):
    """
        Scaled dot-product attention mechanism
    """
    def __init__(self, attention_dropout=0.):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        forward
        :param q: Queries tensor, [B, num_heads, L_q, D_q]
        :param k: Keys tensor, [B, num_heads, L_k, D_k]
        :param v: Values tensor, [B, num_heads, L_v, D_v]
        :param scale: scale factor
        :param attn_mask: Masking tensor, [B, num_heads, L_q, L_k]
        :return:context tensor, attention tensor
        """
        attention = torch.matmul(q, k.transpose(-1, -2))
        if scale:
            attention *= scale
        if attn_mask != None:
            attention = attention.masked_fill(attn_mask, float('-inf'))
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        return context, attention

class MultiHeadAttention(nn.Module):
    """
        Multi-head attention mechanism
    """
    def __init__(self, model_dim=512, out_dim=512, num_heads=8, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.out_dim = out_dim

        self.linear_k = nn.Linear(model_dim, self.dim_per_head*num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head*num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head*num_heads)

        self.dot_product_attention = ScaleDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, key, value, query, attn_mask=None):
        # residual connection
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
        query = query.view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)

        if attn_mask != None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        # scaled dot product attention
        scale = dim_per_head ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask
        )

        # concat heads
        context = context.transpose(1, 2).reshape(batch_size, -1, dim_per_head*num_heads)

        # linear
        output = self.linear_final(context)
        output = self.dropout(output)

        # add residual and norm layer
        if self.model_dim == self.out_dim:
            output += residual
        output = self.layer_norm(output)

        return output, attention


