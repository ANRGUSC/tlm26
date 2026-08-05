"""
RWKV-7 ("x070") language model, drop-in for this repo's training + bpb harness.

Implements the RWKV-v7 recurrence from BlinkDL/RWKV-LM (RWKV_Tmix_x070 /
RWKV_CMix_x070). Uses LayerNorm (pre-norm) rather than RMSNorm, an initial
ln0 on the embeddings and a final ln_out before the tied head. Exposes the
SAME forward contract as model.Transformer:

    forward(tokens, targets=None) -> logits
        - sets self.last_loss = F.cross_entropy(..., ignore_index=-1) when targets given
        - returns logits[:, [-1], :] (last position only) when targets is None
        - self.tok_embeddings.weight is tied to self.output.weight

The per-timestep WKV7 scan is implemented with batched matmuls (correctness
over speed). It is run in float32 for numerical stability of the recurrence.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from model import ModelArgs, Transformer


@dataclass
class RWKVModelArgs(ModelArgs):
    head_size: int = 32
    ffn_dim: Optional[int] = None

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Build tolerantly from an arbitrary dict (e.g. train.py's model_args),
        ignoring any keys that are not fields of this dataclass."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in kwargs.items() if k in fields})


def _lora_ranks(dim: int):
    Dw = max(16, dim // 2)
    Da = max(16, dim // 2)
    Dv = max(16, dim // 4)
    Dg = max(32, dim)
    return Dw, Da, Dv, Dg


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, layer_id: int, args: RWKVModelArgs):
        super().__init__()
        self.layer_id = layer_id
        C = args.dim
        N = args.head_size
        assert C % N == 0, f"dim ({C}) must be divisible by head_size ({N})"
        H = C // N
        self.n_head, self.head_size = H, N
        Dw, Da, Dv, Dg = _lora_ranks(C)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # token-shift mix vectors, each (1,1,C)
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # decay LoRA:  w = -softplus(-(w0 + tanh(xw@w1)@w2)) - 0.5
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, Dw))
        self.w2 = nn.Parameter(torch.empty(Dw, C))

        # in-context learning rate LoRA:  a = sigmoid(a0 + (xa@a1)@a2)
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, Da))
        self.a2 = nn.Parameter(torch.empty(Da, C))

        # gate LoRA:  g = sigmoid(xg@g1)@g2
        self.g1 = nn.Parameter(torch.empty(C, Dg))
        self.g2 = nn.Parameter(torch.empty(Dg, C))

        # value residual mix LoRA (not used on layer 0)
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, C))
            self.v1 = nn.Parameter(torch.empty(C, Dv))
            self.v2 = nn.Parameter(torch.empty(Dv, C))

        # key modulation vectors
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        # r_k bonus term, (H, N)
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # GroupNorm over heads, applied to the (B*T, C) time-mix output
        self.ln_x = nn.GroupNorm(H, C, eps=1e-5)

        self._reset_parameters()

    def _reset_parameters(self):
        # Mix vectors: mild mixing, stable at init.
        for p in (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g):
            nn.init.constant_(p, 0.5)
        # decay bias so w_exp = exp(-exp(w)) is a sensible ~0.7 at init
        nn.init.zeros_(self.w0)
        nn.init.zeros_(self.a0)
        nn.init.constant_(self.k_k, 1.0)
        nn.init.constant_(self.k_a, 1.0)
        nn.init.zeros_(self.r_k)                 # bonus term off at init
        # LoRA "in" matrices small, "out" matrices zero -> a/w/g/v start at their bias
        for p in (self.w1, self.a1, self.g1):
            nn.init.normal_(p, mean=0.0, std=0.02)
        for p in (self.w2, self.a2, self.g2):
            nn.init.zeros_(p)
        if self.layer_id > 0:
            nn.init.zeros_(self.v0)
            nn.init.normal_(self.v1, mean=0.0, std=0.02)
            nn.init.zeros_(self.v2)
        nn.init.normal_(self.receptance.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.weight)       # residual-safe: block ~identity at init

    @staticmethod
    def _wkv7(r, w, k, v, a, b, H, N):
        """Per-timestep linear-attention scan. Inputs (B,T,C); returns (B,T,C).
        Run in float32. Recurrence (state S is (B,H,N,N), init 0):
            w_exp = exp(-exp(w))
            S = S*w_exp + (S@a)@b^T + v@k^T          (all terms use the *old* S)
            out_t = S @ r
        """
        B, T, C = r.shape
        dtype_in = r.dtype
        r = r.float().view(B, T, H, N)
        w = w.float().view(B, T, H, N)
        k = k.float().view(B, T, H, N)
        v = v.float().view(B, T, H, N)
        a = a.float().view(B, T, H, N)
        b = b.float().view(B, T, H, N)
        w_exp = torch.exp(-torch.exp(w))                      # (B,T,H,N)

        S = torch.zeros(B, H, N, N, dtype=torch.float32, device=r.device)
        out = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)
        for t in range(T):
            wt = w_exp[:, t]                                  # (B,H,N)
            at = a[:, t].unsqueeze(-1)                        # (B,H,N,1)
            bt = b[:, t].unsqueeze(-2)                        # (B,H,1,N)
            kt = k[:, t].unsqueeze(-2)                        # (B,H,1,N)
            vt = v[:, t].unsqueeze(-1)                        # (B,H,N,1)
            rt = r[:, t].unsqueeze(-1)                        # (B,H,N,1)
            # all three terms reference the OLD S (per the x070 spec line)
            S = (S * wt[:, :, None, :]
                 + torch.matmul(torch.matmul(S, at), bt)
                 + torch.matmul(vt, kt))
            out[:, t] = torch.matmul(S, rt).squeeze(-1)       # (B,H,N)
        return out.view(B, T, C).to(dtype_in)

    def forward(self, x, v_first):
        B, T, C = x.shape
        H, N = self.n_head, self.head_size

        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # WKV7 op:  a_param = -kk,  b_param = kk*a
        out = self._wkv7(r, w, k, v, -kk, kk * a, H, N)

        # GroupNorm over heads
        out = self.ln_x(out.view(B * T, C)).view(B, T, C)

        # x070 r_k bonus term
        rr = r.view(B, T, H, N)
        kkk = k.view(B, T, H, N)
        vv = v.view(B, T, H, N)
        bonus = ((rr * kkk * self.r_k).sum(dim=-1, keepdim=True) * vv).view(B, T, C)
        out = out + bonus

        out = self.output(out * g)
        return out, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, layer_id: int, args: RWKVModelArgs):
        super().__init__()
        C = args.dim
        ffn = args.ffn_dim if args.ffn_dim is not None else 4 * C
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.key = nn.Linear(C, ffn, bias=False)
        self.value = nn.Linear(ffn, C, bias=False)
        nn.init.constant_(self.x_k, 0.5)
        nn.init.normal_(self.key.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value.weight)        # residual-safe at init

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.x_k
        return self.value(torch.relu(self.key(xk)) ** 2)


class RWKVBlock(nn.Module):
    def __init__(self, layer_id: int, args: RWKVModelArgs):
        super().__init__()
        self.ln1 = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ln2 = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.att = RWKV_Tmix_x070(layer_id, args)
        self.ffn = RWKV_CMix_x070(layer_id, args)

    def forward(self, x, v_first):
        att_out, v_first = self.att(self.ln1(x), v_first)
        x = x + att_out
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKVTransformer(Transformer):
    """RWKV-7 model with the same forward contract as model.Transformer.
    Reuses Transformer.configure_optimizers / generate / estimate_mfu."""
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: RWKVModelArgs):
        nn.Module.__init__(self)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.drop = nn.Dropout(params.dropout)
        self.ln0 = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.blocks = nn.ModuleList([RWKVBlock(i, params) for i in range(params.n_layers)])
        self.ln_out = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # weight tying (embedding <-> unembedding)
        self.tok_embeddings.weight = self.output.weight

        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        self.oe_size = 0
        self.last_loss = None

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  RWKV-7 params: {n_params:,} total "
              f"(dim={params.dim}, n_layers={params.n_layers}, head_size={params.head_size}, "
              f"H={params.dim // params.head_size})")

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        x = self.tok_embeddings(tokens)
        x = self.drop(x)
        x = self.ln0(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)

        if targets is not None:
            logits = self.output(x)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(x[:, [-1], :])   # last position only
            self.last_loss = None
        return logits
