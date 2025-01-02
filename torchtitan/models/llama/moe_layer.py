# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        logits = logits.view(-1, self.num_experts)
        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        scores = torch.softmax(top_k_values, dim=-1, dtype=torch.float32).type_as(x)
        tokens_per_expert = torch.bincount(
            top_k_indices.view(-1), minlength=self.num_experts
        )
        return scores, top_k_indices, tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class TokenDispatcher:
    def __init__(self, top_k: int):
        self.top_k = top_k
        self.hidden_state_shape = None
        self.reversed_input_permutation_mapping = None

    def token_permutation(self, x: torch.Tensor, indices: torch.Tensor):
        self.hidden_state_shape = x.shape
        x = x.view(-1, x.size(-1))
        flatten_indices = indices.flatten()
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        permuted_tokens = x.index_select(0, sorted_indices // self.top_k)
        self.reversed_input_permutation_mapping = sorted_indices
        return permuted_tokens

    def token_unpermutation(
        self, permuted_tokens: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        num_unpermuted_tokens = scores.numel()
        unpermuted_tokens = torch.zeros(
            (num_unpermuted_tokens, permuted_tokens.size(1)),
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_copy_(
            0, self.reversed_input_permutation_mapping, permuted_tokens
        )
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, self.top_k, permuted_tokens.size(1)
        )

        unpermuted_tokens = unpermuted_tokens * scores.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1).type_as(permuted_tokens)
        output = unpermuted_tokens.view(self.hidden_state_shape)
        return output


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class Experts(nn.Module):
    def __init__(self, dim: int, moe_expert_dim: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [FeedForward(dim, moe_expert_dim, 1, 1) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        tokens_per_expert = tokens_per_expert.tolist()
        tokens_list = torch.split(x, tokens_per_expert, dim=0)
        expert_outputs = [
            expert(tokens) for expert, tokens in zip(self.experts, tokens_list)
        ]
        return torch.cat(expert_outputs, dim=0)

    def init_weights(self, init_std: float):
        for expert in self.experts:
            expert.init_weights(init_std)


class MoELayer(nn.Module):
    def __init__(self, dim: int, moe_expert_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = TopKRouter(dim, num_experts, top_k)
        self.dispatcher = TokenDispatcher(top_k)
        self.experts = Experts(dim, moe_expert_dim, num_experts)

    def forward(self, x: torch.Tensor):
        scores, top_k_indices, tokens_per_expert = self.router(x)
        permuted_tokens = self.dispatcher.token_permutation(x, top_k_indices)
        expert_outputs = self.experts(permuted_tokens, tokens_per_expert)
        return self.dispatcher.token_unpermutation(expert_outputs, scores)

    def init_weights(self, init_std: float):
        self.router.init_weights(init_std)
        self.experts.init_weights(init_std)


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = torch.device("cuda")
    x = torch.randn(2, 2048, 4096, dtype=dtype, device=device)
    moe_layer = MoELayer(4096, 1024, 8, 2).to(device).to(dtype)
    print(moe_layer)
    output = moe_layer(x)
    print(output.shape)
