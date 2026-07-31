# Every-model scaling figure: bits/byte vs parameters, offset power-law fit L = E + A*N^-a.
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

GROUP = {
    "width": "capacity", "depth": "capacity", "size_ladder": "capacity",
    "shape": "shape", "tokenizer": "tokenizer", "vocab": "tokenizer",
    "quant": "quant", "data_shards": "data", "data_tokens": "data",
    "looped": "looped", "baseline": "baseline",
    "distill": "other", "data_quality": "other", "rag": "other",
}
STYLE = {
    "capacity":  ("o", "Capacity (width / depth / size)"),
    "shape":     ("s", "Shape (iso-param)"),
    "tokenizer": ("^", "Tokenizer / vocab"),
    "quant":     ("D", "Quantization (QAT)"),
    "data":      ("P", "Data quantity"),
    "looped":    ("X", "Looped"),
    "other":     (".", "Other (distill / GPT-4 data)"),
    "baseline":  ("*", "Released baselines"),
}
ORDER = ["capacity", "shape", "tokenizer", "quant", "data", "looped", "other", "baseline"]
FIT_DIRS = ["out_width_d32_long", "out_width_d48_long", "out_width_d64_long",
            "out_width_d96_long", "out_width_d112_long", "out_width_d128_long",
            "out_width_d160_long", "out_width_d224_long"]


def load():
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["bpb"] = float(r["bits_per_byte"]) if r["bits_per_byte"] else None
        r["N"] = int(r["params_total"]) if r["params_total"] else None
        r["grp"] = GROUP.get(r["experiment"], "other")
    return rows


def grid_fit(N, L):
    logN = np.log(N)
    best = None
    for E in np.linspace(0.0, L.min() * 0.999, 600):
        y = L - E
        if np.any(y <= 0):
            continue
        b, la = np.polyfit(logN, np.log(y), 1)
        A, al = np.exp(la), -b
        yh = E + A * np.power(N, -al)
        r2 = 1 - np.sum((L - yh) ** 2) / np.sum((L - L.mean()) ** 2)
        if best is None or r2 > best[-1]:
            best = (E, A, al, r2)
    return best


def offset_power_fit(N, L):
    N, L = np.asarray(N, float), np.asarray(L, float)
    try:
        from scipy.optimize import curve_fit
        f = lambda n, E, A, al: E + A * np.power(n, -al)
        popt, _ = curve_fit(f, N, L, p0=[0.8 * L.min(), 1.0, 0.3],
                            bounds=([0, 0, 0], [L.min(), np.inf, 2.0]), maxfev=200000)
        E, A, al = popt
    except Exception:
        E, A, al, _ = grid_fit(N, L)
    yh = E + A * np.power(N, -al)
    r2 = 1 - np.sum((L - yh) ** 2) / np.sum((L - L.mean()) ** 2)
    return E, A, al, r2


def main():
    rows = load()
    by_dir = {r["checkpoint_dir"]: r for r in rows}
    fit = [d for d in FIT_DIRS if d in by_dir and by_dir[d]["bpb"]]   # skip not-yet-trained dirs
    Nf = np.array([by_dir[d]["N"] for d in fit], float)
    Lf = np.array([by_dir[d]["bpb"] for d in fit], float)
    o = np.argsort(Nf); Nf, Lf = Nf[o], Lf[o]
    E, A, al, r2 = offset_power_fit(Nf, Lf)
    print(f"L = {E:.4f} + {A:.4f} * N^-{al:.4f}   R2 = {r2:.4f}")

    plt.figure(figsize=(9, 6))
    for g in ORDER:
        mk, lab = STYLE[g]
        pts = [r for r in rows if r["grp"] == g and r["bpb"] and r["N"]
               and (r["converged"] == "1" or g == "baseline")]
        if not pts:
            continue
        size = 130 if g == "baseline" else 45
        plt.scatter([r["N"] for r in pts], [r["bpb"] for r in pts], marker=mk, s=size, label=lab)

    xin = np.logspace(np.log10(Nf.min()), np.log10(Nf.max()), 200)
    xext = np.logspace(np.log10(Nf.max()), np.log10(1.05e7), 200)
    plt.plot(xin, E + A * np.power(xin, -al), color="red", linewidth=2, label="power-law fit")
    plt.plot(xext, E + A * np.power(xext, -al), color="red", linestyle="--", linewidth=2)

    plt.text(1.0e5, 0.70, f"bpb = {E:.3f} + {A:.1f}·N$^{{-{al:.3f}}}$\nR² = {r2:.4f}", fontsize=9)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Bits Per Byte")
    plt.title("Bits Per Byte vs Parameters")
    plt.legend(fontsize=8)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "scaling_figure.png"), dpi=300)
    plt.savefig(os.path.join(FIG, "scaling_figure.svg"))
    plt.close()
    print("wrote scaling_figure.png/.svg")


if __name__ == "__main__":
    main()
