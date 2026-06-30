"""Stage 4: MEASURED gamma-sweep for the best config. Runs the real decoder at
each gamma on every prompt, records per-prompt wall-clock speedup vs baseline,
and reports mean + bootstrap 95% CI. This validates the predicted speedup curve
across all of gamma, not just one point. Writes data/sweep.json.

Usage: /usr/bin/python3 scripts/04_sweep.py 1.5B 7B
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import decode, prompts, stats

draft  = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
target = sys.argv[2] if len(sys.argv) > 2 else "7B"
GAMMAS = [1, 2, 3, 4, 5, 6]
MAX_NEW = 96
test_prompts = prompts.load_code(60)[:30]

# warmup
decode.baseline_greedy(target, "Say hello.", max_new=8)
decode.speculative_greedy(draft, target, "Say hello.", max_new=8, gamma=3)

# baseline once per prompt (reused across gammas)
print(f"baseline {target} on {len(test_prompts)} prompts...")
base = [decode.baseline_greedy(target, p, max_new=MAX_NEW) for p in test_prompts]

out = {"draft": draft, "target": target, "gammas": GAMMAS, "by_gamma": {}}
print(f"\nmeasured gamma-sweep ({draft}->{target}):")
for g in GAMMAS:
    speedups, accept_rates, accept_lens, exact_n = [], [], [], 0
    for p, b in zip(test_prompts, base):
        s = decode.speculative_greedy(draft, target, p, max_new=MAX_NEW, gamma=g)
        speedups.append(b["time"] / s["time"])
        accept_rates.append(s["accept_rate"])
        accept_lens.append(s["mean_accept_len"])
        exact_n += int(b["tokens"] == s["tokens"])
    mean, (lo, hi) = stats.bootstrap_ci(speedups)
    ar, _ = stats.bootstrap_ci(accept_rates)
    al, _ = stats.bootstrap_ci(accept_lens)
    out["by_gamma"][g] = {"speedup": mean, "ci": [lo, hi], "accept_rate": ar,
                          "accept_len": al, "exact": exact_n, "n": len(test_prompts)}
    print(f"  gamma={g}: speedup {mean:.2f}x [{lo:.2f},{hi:.2f}]  "
          f"accept={ar:.2f}  len={al:.2f}  exact={exact_n}/{len(test_prompts)}")

best_g = max(out["by_gamma"], key=lambda g: out["by_gamma"][g]["speedup"])
out["best_gamma"] = best_g
print(f"\nmeasured optimum: gamma={best_g} "
      f"({out['by_gamma'][best_g]['speedup']:.2f}x)")

with open("data/sweep.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote data/sweep.json")
