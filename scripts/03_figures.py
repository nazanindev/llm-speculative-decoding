"""Stage 3: figures from data/measurements.json.

  fig1_where_it_pays  -- predicted speedup for every (draft,target,domain);
                         the y=1 line is the break-even (below it, spec LOSES).
  fig2_gamma_sweep     -- predicted speedup vs gamma for the best config,
                         with the measured wall-clock point overlaid.
"""
import os, sys, json
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import theory

M = json.load(open("data/measurements.json"))
os.makedirs("figures", exist_ok=True)

# ---- fig 1: where speculation pays off ----
keys = list(M["predicted"].keys())
sp = [M["predicted"][k]["speedup"] for k in keys]
colors = ["tab:green" if s >= 1 else "tab:red" for s in sp]
plt.figure(figsize=(9, 5))
plt.bar(range(len(keys)), sp, color=colors)
plt.axhline(1.0, ls="--", c="black", lw=1)
plt.xticks(range(len(keys)), keys, rotation=30, ha="right")
plt.ylabel("predicted speedup vs autoregressive")
plt.title("When does speculative decoding pay off? (green = win, red = loss)")
plt.tight_layout(); plt.savefig("figures/fig1_where_it_pays.png", dpi=150); plt.close()

# ---- fig 2: gamma sweep for the best config ----
best_key = max(keys, key=lambda k: M["predicted"][k]["speedup"])
alpha = M["alpha"][best_key]; c = M["predicted"][best_key]["c"]
gammas = list(range(1, 11))
curve = [theory.speedup(alpha, g, c) for g in gammas]
plt.figure(figsize=(8, 5))
plt.plot(gammas, curve, marker="o", label=f"predicted ({best_key}, α={alpha:.2f}, c={c:.2f})")
# measured point from stage 2 (1.5B->7B/code, gamma=3): 1.32x
plt.scatter([3], [1.32], color="tab:red", zorder=5, s=80, label="measured wall-clock (γ=3)")
plt.axhline(1.0, ls="--", c="gray")
plt.xlabel("gamma (draft tokens per iteration)"); plt.ylabel("speedup")
plt.title("Speedup vs draft length: an optimum, then diminishing returns")
plt.legend(); plt.tight_layout(); plt.savefig("figures/fig2_gamma_sweep.png", dpi=150); plt.close()

print("best config:", best_key, "->", round(M["predicted"][best_key]["speedup"], 2), "x predicted")
print("wrote figures/fig1_where_it_pays.png, figures/fig2_gamma_sweep.png")
