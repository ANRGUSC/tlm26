# Single-variable ablation plots (width, depth, shape, tokenizer, quant, data) at the 30k budget.
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "master_data.csv")
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

WIDTH = ["out_width_d32_long", "out_width_d48_long", "out_width_d64_long",
         "out_width_d96_long", "out_width_d128_long"]
DEPTH = ["out_depth_L1_long", "out_depth_L2_long", "out_depth_L3_long", "out_depth_L4_long",
         "out_width_d64_long", "out_depth_L6_long", "out_depth_L8_long"]
SHAPE = ["out_iso_d96L2_long", "out_iso_d80L3_long", "out_width_d64_long", "out_iso_d48L10_long"]
TOKENIZER = [("out_char105_long", "char (105)"), ("out_tok_256_long", "BPE-256"),
             ("out_width_d64_long", "BPE-512"), ("out_tok_512dep_long", "BPE-512 dep"),
             ("out_tok_512nf_long", "BPE-512 nf"), ("out_tok_512uni_long", "unigram-512"),
             ("out_B_long", "BPE-1024"), ("out_C_long", "BPE-2048")]
QUANT = [("out_width_d64_long", "fp32 d64"), ("out_qat_int4_long", "int4 d64"),
         ("out_qat_ternary_long", "ternary d64"), ("out_qat_ternary_d96_long", "ternary d96")]
DATA_T = [("out_data_t300k_long", 0.3), ("out_data_t1m_long", 1.0), ("out_data_t3m_long", 3.0),
          ("out_data_t10m_long", 10.0), ("out_width_d64_long", 1850.0)]


def load():
    with open(CSV, encoding="utf-8") as f:
        return {r["checkpoint_dir"]: r for r in csv.DictReader(f)}


def ok(r):
    return bool(r) and r["bits_per_byte"] and r["converged"] == "1"


def label_pts(x, y, fmt="{:.3f}", va="bottom"):
    for xi, yi in zip(x, y):
        plt.text(xi, yi, fmt.format(yi), fontsize=9, ha="center", va=va)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=300)
    plt.close()
    print(f"  wrote {name}")


def plot_width(rows):
    pts = [rows[d] for d in WIDTH if ok(rows.get(d))]
    if len(pts) < 3:
        print("  width: skip"); return
    N = np.array([int(r["params_total"]) for r in pts], float)
    y = np.array([float(r["bits_per_byte"]) for r in pts], float)
    b, la = np.polyfit(np.log(N), np.log(y), 1); a = np.exp(la)
    r2 = 1 - np.sum((y - a * N ** b) ** 2) / np.sum((y - y.mean()) ** 2)
    plt.figure(figsize=(7, 5))
    xs = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
    plt.plot(xs, a * xs ** b, color="red", linestyle="--", linewidth=2,
             label=f"power-law fit (R²={r2:.4f})")
    plt.plot(N, y, marker="o", linewidth=2, label="bits/byte")
    label_pts(N, y)
    plt.xscale("log")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Bits Per Byte")
    plt.title("Bits Per Byte vs Width")
    plt.legend()
    plt.grid(True, which="both")
    save("ablation_width.png")


def plot_depth(rows):
    pts = [rows[d] for d in DEPTH if ok(rows.get(d))]
    if len(pts) < 3:
        print("  depth: skip"); return
    L = np.array([int(r["n_layers"]) for r in pts], float)
    y = np.array([float(r["bits_per_byte"]) for r in pts], float)
    o = np.argsort(L); L, y = L[o], y[o]
    plt.figure(figsize=(7, 5))
    plt.plot(L, y, marker="o", linewidth=2, label="bits/byte")
    label_pts(L, y)
    plt.axvline(3, color="red", linestyle="--", linewidth=2, label="elbow ≈ 3 layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("Bits Per Byte")
    plt.title("Bits Per Byte vs Depth")
    plt.xticks(sorted(int(v) for v in L))
    plt.legend()
    plt.grid(True)
    save("ablation_depth.png")


def plot_shape(rows):
    pts = [rows[d] for d in SHAPE if ok(rows.get(d))]
    if len(pts) < 3:
        print("  shape: skip"); return
    L = np.array([int(r["n_layers"]) for r in pts], float)
    y = np.array([float(r["bits_per_byte"]) for r in pts], float)
    dim = [int(r["dim"]) for r in pts]
    o = np.argsort(L); L, y = L[o], y[o]; dim = [dim[i] for i in o]
    plt.figure(figsize=(7, 5))
    plt.plot(L, y, marker="o", linewidth=2, label="bits/byte (~279K fixed)")
    for xi, yi, di in zip(L, y, dim):
        plt.text(xi, yi, f"d{di}: {yi:.3f}", fontsize=9, ha="center", va="bottom")
    plt.xlabel("Number of Layers")
    plt.ylabel("Bits Per Byte")
    plt.title("Bits Per Byte vs Shape (Fixed Parameters)")
    plt.xticks(sorted(int(v) for v in L))
    plt.legend()
    plt.grid(True)
    save("ablation_shape.png")


def bar_chart(rows, spec, title, xlabel, name):
    items = [(lab, rows.get(d)) for d, lab in spec]
    items = [(lab, r) for lab, r in items if ok(r)]
    if len(items) < 3:
        print(f"  {name}: skip"); return
    labs = [lab for lab, _ in items]
    y = [float(r["bits_per_byte"]) for _, r in items]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(range(len(labs)), y)
    for i, yi in enumerate(y):
        plt.text(i, yi, f"{yi:.3f}", fontsize=9, ha="center", va="bottom")
    plt.xticks(range(len(labs)), labs, rotation=30, ha="right", fontsize=9)
    plt.xlabel(xlabel)
    plt.ylabel("Bits Per Byte")
    plt.title(title)
    plt.grid(axis="y")
    save(name)


def plot_data(rows):
    pts = [(mt, rows.get(d)) for d, mt in DATA_T]
    pts = [(mt, r) for mt, r in pts if ok(r)]
    if len(pts) < 3:
        print("  data: skip"); return
    pts.sort(key=lambda t: t[0])
    x = np.array([mt for mt, _ in pts]); y = np.array([float(r["bits_per_byte"]) for _, r in pts])
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o", linewidth=2, label="bits/byte")
    label_pts(x, y)
    plt.axvline(1, color="red", linestyle="--", linewidth=2, label="floor ≈ 1M tokens")
    plt.xscale("log")
    plt.xlabel("Training Tokens (Millions)")
    plt.ylabel("Bits Per Byte")
    plt.title("Bits Per Byte vs Training Data")
    plt.legend()
    plt.grid(True, which="both")
    save("ablation_data.png")


def main():
    rows = load()
    print("WIDTH:"); plot_width(rows)
    print("DEPTH:"); plot_depth(rows)
    print("SHAPE:"); plot_shape(rows)
    print("TOKENIZER:")
    bar_chart(rows, TOKENIZER, "Bits Per Byte by Tokenizer", "Tokenizer (Vocabulary)",
              "ablation_tokenizer.png")
    print("QUANT:")
    bar_chart(rows, QUANT, "Bits Per Byte by Quantization Precision", "Weight Precision",
              "ablation_quant.png")
    print("DATA:"); plot_data(rows)


if __name__ == "__main__":
    main()
