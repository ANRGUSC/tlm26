# Chat (QA-tuned) accuracy by base model size. Numbers from rag_poc/eval_qa.py on 600 held-out QA.
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

labels = ["d32 (78K)", "d48 (152K)", "d64 (279K)", "d96 (557K)", "d128 (989K)"]
all_em = [0.760, 0.825, 0.905, 0.958, 0.972]
rare_em = [0.000, 0.023, 0.349, 0.605, 0.744]

x = np.arange(len(labels))
w = 0.38
plt.figure(figsize=(7, 5))
b1 = plt.bar(x - w / 2, all_em, w, label="All questions")
b2 = plt.bar(x + w / 2, rare_em, w, label="Rare names (must copy from context)")
for bars in (b1, b2):
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{bar.get_height():.3f}", fontsize=9, ha="center", va="bottom")

plt.xticks(x, labels)
plt.ylim(0, 1.18)
plt.xlabel("Chat Model (Base Size)")
plt.ylabel("Exact-Match Accuracy")
plt.title("Chat QA Accuracy by Model Size")
plt.legend(loc="upper center", ncol=2, fontsize=9)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "chat_compare.png"), dpi=300)
plt.close()
print("wrote chat_compare.png")
