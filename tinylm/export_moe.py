# Export a trained MoE checkpoint to an int8 blob for the C engine (llama2_moe_core.h).
# Router (gate) kept fp32 (tiny, preserves expert selection); experts int8, group-quantized.
# Layout after a 256B header: fp32 [attn_norms, ffn_norms, final_norm, gates] then int8
# [tok_emb, wq, wk, wv, wo, w1e(L*E), w2e(L*E), w3e(L*E)] each as (int8 q, fp32 scales).
import os
import struct
import sys

import torch

from export import quantize_q80, serialize_fp32, serialize_int8
from model_moe import MoETransformer, MoEModelArgs

MAGIC = 0x616B4D31  # "akM1"


def load_moe(ckpt_dir):
    ck = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    ma = ck["model_args"]
    m = MoETransformer(MoEModelArgs(**ma))
    sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    m.load_state_dict(sd)
    m.eval()
    return m, ma


def export(ckpt_dir, out_path, group_size=64):
    m, ma = load_moe(ckpt_dir)
    p = m.params
    L, E, K = p.n_layers, ma["n_experts"], ma["moe_top_k"]
    dim = p.dim
    hidden = m.layers[0].feed_forward.experts[0].w1.weight.shape[0]
    n_kv = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    shared = torch.equal(m.tok_embeddings.weight, m.output.weight)
    while dim % group_size != 0:
        group_size //= 2

    # ordered int8 weights
    int8_w = [m.tok_embeddings.weight]
    int8_w += [l.attention.wq.weight for l in m.layers]
    int8_w += [l.attention.wk.weight for l in m.layers]
    int8_w += [l.attention.wv.weight for l in m.layers]
    int8_w += [l.attention.wo.weight for l in m.layers]
    int8_w += [l.feed_forward.experts[e].w1.weight for l in m.layers for e in range(E)]
    int8_w += [l.feed_forward.experts[e].w2.weight for l in m.layers for e in range(E)]
    int8_w += [l.feed_forward.experts[e].w3.weight for l in m.layers for e in range(E)]
    if not shared:
        int8_w.append(m.output.weight)
    for w in int8_w:
        assert w.numel() % group_size == 0, f"{tuple(w.shape)} numel not multiple of {group_size}"

    f = open(out_path, "wb")
    f.write(struct.pack("I", MAGIC))
    f.write(struct.pack("i", 2))
    f.write(struct.pack("iiiiiii", dim, hidden, L, p.n_heads, n_kv, p.vocab_size, p.max_seq_len))
    f.write(struct.pack("ii", E, K))
    f.write(struct.pack("B", int(shared)))
    f.write(struct.pack("i", group_size))
    f.write(b"\0" * (256 - f.tell()))

    # fp32 block: norms + router gates
    for l in m.layers:
        serialize_fp32(f, l.attention_norm.weight)
    for l in m.layers:
        serialize_fp32(f, l.ffn_norm.weight)
    serialize_fp32(f, m.norm.weight)
    for l in m.layers:
        serialize_fp32(f, l.feed_forward.gate.weight)   # (E, dim)

    # int8 block
    maxerr = 0.0
    for w in int8_w:
        q, s, err = quantize_q80(w, group_size)
        serialize_int8(f, q)
        serialize_fp32(f, s)
        maxerr = max(maxerr, err)
    f.close()

    sz = os.path.getsize(out_path)
    print(f"wrote {out_path}  {sz/1e6:.2f} MB  dim={dim} hidden={hidden} L={L} E={E} K={K} "
          f"vocab={p.vocab_size} GS={group_size} shared={shared} maxerr={maxerr:.4f}")


if __name__ == "__main__":
    ckpt_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else ckpt_dir.rstrip("/\\") + ".moe.bin"
    export(ckpt_dir, out_path)
