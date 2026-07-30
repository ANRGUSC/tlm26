"""
Looped Transformer variant of llama2.c model.
Instead of N unique layers, uses K unique layers looped L times (effective depth = K*L).
Based on "Simply Stabilizing the Loop via Fully Looped Transformer" (arxiv 2605.18797).
"""
import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model import (
    RMSNorm, Attention, FeedForward, TransformerBlock,
    precompute_freqs_cis, ModelArgs,
)


@dataclass
class LoopedModelArgs(ModelArgs):
    n_unique_layers: int = 2
    n_loops: int = 3
    # Mythos-style input injection: re-add the input embedding at each loop iteration.
    # Default OFF — our data found injection adds noise (Loop 2x3+inject underperformed
    # plain Loop 2x3). Stored in model_args so eval reproduces the trained behavior.
    input_injection: bool = False


class LoopedTransformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: LoopedModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_unique_layers = params.n_unique_layers
        self.n_loops = params.n_loops
        self.input_injection = params.input_injection
        self.effective_depth = params.n_unique_layers * params.n_loops

        # residual scaling factor (1/sqrt(n_loops)) for training stability
        self.residual_scale = 1.0 / self.n_loops

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)

        # only create n_unique_layers, NOT n_layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_unique_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # weight tying
        self.tok_embeddings.weight = self.output.weight

        # RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            params.dim // params.n_heads, params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.effective_depth))

        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # save input embedding for injection (Mythos-style)
        input_embed = h

        # loop through the unique layers n_loops times
        for loop_idx in range(self.n_loops):
            # optional input-embedding injection at each loop iteration (default off)
            if self.input_injection and loop_idx > 0:
                h = h + input_embed
            for layer in self.layers:
                # residual scaling: scale the layer output before adding to residual
                residual = h
                h = layer(h, freqs_cos, freqs_sin)
                # apply residual scaling: h = residual + scale * (h - residual)
                h = residual + self.residual_scale * (h - residual)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L = self.effective_depth
        H, Q, T = cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
