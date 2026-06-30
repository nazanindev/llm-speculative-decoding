"""Stage 3: figures from data/measurements.json (+ data/sweep.json if present).

  fig1_where_it_pays  -- predicted speedup for every (draft,target,domain),
                         break-even line at y=1.
  fig2_gamma_sweep     -- predicted speedup vs gamma for the best config,
                         with the MEASURED wall-clock curve (bootstrap CIs) over it.
  fig3_alpha_domain    -- acceptance rate by config, code vs prose, with CIs.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import theory

M = json.load(open("data/measurements.json"))
SWEEP = json.load(open("data/sweep.json")) if os.path.exists("data/sweep.json") else None
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

# ---- fig 2: predicted vs MEASURED gamma sweep (same config on both curves) ----
best_key = (f"{SWEEP['draft']}->{SWEEP['target']}/code" if SWEEP
            else max(keys, key=lambda k: M["predicted"][k]["speedup"]))
alpha = M["alpha"][best_key]; c = M["predicted"][best_key]["c"]
gammas = list(range(1, 11))
curve = [theory.speedup(alpha, g, c) for g in gammas]
plt.figure(figsize=(8, 5))
plt.plot(gammas, curve, marker="o", color="tab:blue",
         label=f"predicted ({best_key}, α={alpha:.2f}, c={c:.2f})")
if SWEEP:
    mg = SWEEP["gammas"]
    mv = [SWEEP["by_gamma"][str(g)]["speedup"] for g in mg]
    err = np.array([[SWEEP["by_gamma"][str(g)]["speedup"] - SWEEP["by_gamma"][str(g)]["ci"][0],
                     SWEEP["by_gamma"][str(g)]["ci"][1] - SWEEP["by_gamma"][str(g)]["speedup"]]
                    for g in mg]).T
    plt.errorbar(mg, mv, yerr=err, marker="s", capsize=4, color="tab:red",
                 label=f"measured wall-clock ({SWEEP['draft']}->{SWEEP['target']}, 95% CI)")
plt.axhline(1.0, ls="--", c="gray")
plt.xlabel("gamma (draft tokens per iteration)"); plt.ylabel("speedup")
plt.title("Speedup vs draft length: predicted vs measured")
plt.legend(); plt.tight_layout(); plt.savefig("figures/fig2_gamma_sweep.png", dpi=150); plt.close()

# ---- fig 3: acceptance rate by config, code vs prose ----
pairs = ["0.5B->3B", "1.5B->3B", "0.5B->7B", "1.5B->7B"]
code_a = [M["alpha"][f"{p}/code"] for p in pairs]
prose_a = [M["alpha"][f"{p}/prose"] for p in pairs]
code_e = np.array([[M["alpha"][f"{p}/code"] - M["alpha_ci"][f"{p}/code"][0],
                    M["alpha_ci"][f"{p}/code"][1] - M["alpha"][f"{p}/code"]] for p in pairs]).T
prose_e = np.array([[M["alpha"][f"{p}/prose"] - M["alpha_ci"][f"{p}/prose"][0],
                     M["alpha_ci"][f"{p}/prose"][1] - M["alpha"][f"{p}/prose"]] for p in pairs]).T
x = np.arange(len(pairs)); w = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - w/2, code_a, w, yerr=code_e, capsize=4, label="code", color="tab:blue")
plt.bar(x + w/2, prose_a, w, yerr=prose_e, capsize=4, label="prose", color="tab:orange")
plt.xticks(x, pairs); plt.ylim(0, 1.0); plt.ylabel("acceptance rate α (95% CI)")
plt.title("Acceptance is domain-dependent: code >> prose")
plt.legend(); plt.tight_layout(); plt.savefig("figures/fig3_alpha_domain.png", dpi=150); plt.close()

print("best config:", best_key, "->", round(M["predicted"][best_key]["speedup"], 2), "x predicted")
if SWEEP:
    print("measured optimum: gamma", SWEEP["best_gamma"],
          round(SWEEP["by_gamma"][str(SWEEP["best_gamma"])]["speedup"], 2), "x")
print("wrote fig1_where_it_pays, fig2_gamma_sweep, fig3_alpha_domain")
