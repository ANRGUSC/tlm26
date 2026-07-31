# Mixture-of-Experts variant: SwiGLU FFN replaced by N experts + top-k router (Switch/Mixtral style).
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from model import ModelArgs, RMSNorm, Attention, precompute_freqs_cis, Transformer


@dataclass
class MoEModelArgs(ModelArgs):
    n_experts: int = 8
    moe_top_k: int = 2
    aux_coef: float = 0.01


class Expert(nn.Module):  # one SwiGLU FFN
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return F.linear(F.silu(F.linear(x, self.w1.weight)) * F.linear(x, self.w3.weight), self.w2.weight)


class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout, n_experts, top_k):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.n_experts, self.top_k = n_experts, top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(n_experts)])
        self.dropout = nn.Dropout(dropout)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        xf = x.reshape(-1, dim)                              # (N, dim)
        probs = F.softmax(self.gate(xf), dim=-1)            # (N, E)
        topw, topi = probs.topk(self.top_k, dim=-1)         # (N, k)
        topw = topw / topw.sum(dim=-1, keepdim=True)        # renormalise the kept weights

        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            tok, kk = (topi == e).nonzero(as_tuple=True)    # tokens routing to expert e (+ which slot)
            if tok.numel() == 0:
                continue
            ye = self.experts[e](xf[tok]) * topw[tok, kk].unsqueeze(-1)
            out.index_add_(0, tok, ye)

        # load-balancing aux loss (Switch): E * sum_e f_e * P_e  (f detached, P carries grad)
        disp = torch.zeros(xf.size(0), self.n_experts, device=xf.device)
        disp.scatter_(1, topi, 1.0)
        f = disp.mean(dim=0)                                # fraction of tokens per expert
        P = probs.mean(dim=0)                               # mean router prob per expert
        self.aux_loss = self.n_experts * (f.detach() * P).sum()

        return self.dropout(out).reshape(bsz, seqlen, dim)


class MoETransformerBlock(nn.Module):
    def __init__(self, layer_id, args: MoEModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MoEFeedForward(args.dim, args.hidden_dim, args.multiple_of,
                                           args.dropout, args.n_experts, args.moe_top_k)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        return h + self.feed_forward.forward(self.ffn_norm(h))


class MoETransformer(Transformer):
    """Dense Transformer with MoE feed-forward blocks. Reuses Transformer's
    configure_optimizers / generate / estimate_mfu."""

    def __init__(self, params: MoEModelArgs):
        nn.Module.__init__(self)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.aux_coef = params.aux_coef

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([MoETransformerBlock(i, params) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight     # weight tying

        self.oe_size = 0
        freqs_cos, freqs_sin = precompute_freqs_cis(params.dim // params.n_heads, params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))
        self.last_loss = None

    def forward(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.dropout(F.embedding(tokens, self.tok_embeddings.weight))
        fc, fs = self.freqs_cos[:seqlen], self.freqs_sin[:seqlen]
        for layer in self.layers:
            h = layer(h, fc, fs)
        h = self.norm(h)

        if targets is not None:
            logits = F.linear(h, self.output.weight)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            aux = sum(l.feed_forward.aux_loss for l in self.layers) / len(self.layers)
            self.last_loss = ce + self.aux_coef * aux
            self.last_ce = ce.detach()                      # pure CE, for honest bpb eval
        else:
            logits = F.linear(h[:, [-1], :], self.output.weight)
            self.last_loss = None
        return logits
