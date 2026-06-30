"""Stage 5: sampled (non-greedy) speculative decoding.

Part A -- correctness. Sampled speculation should produce output distributed
  identically to sampling from the target alone. We sample the first token many
  times from (i) the target baseline and (ii) speculative decoding, and compare
  the empirical distributions by total-variation distance. The honest yardstick
  is the baseline-vs-baseline TV (pure sampling noise at this N): if
  TV(baseline, spec) is in the same ballpark, the algorithm is correct.

Part B -- temperature sweep. Higher temperature flattens the distributions, so
  the draft agrees with the target less often: acceptance (and thus speedup)
  should fall with T. We measure acceptance rate per T (bootstrap CI) and the
  predicted speedup, plus a measured wall-clock speedup at gamma=3.

Usage: /usr/bin/python3 scripts/05_sampled.py 1.5B 7B
"""
import os, sys, json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import decode, prompts, stats, theory

draft  = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
target = sys.argv[2] if len(sys.argv) > 2 else "7B"
GAMMA = 3
MAX_NEW = 96
M = json.load(open("data/measurements.json"))
c = M["latency"][draft] / M["latency"][target]

def tv(a, b):
    """total-variation distance between two token multisets of equal size."""
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    na, nb = len(a), len(b)
    return 0.5 * sum(abs(ca[k]/na - cb[k]/nb) for k in keys)

# ---------- Part A: distributional correctness ----------
print("== correctness: first-token distribution (T=1.0) ==")
cp = "Write a Python function to compute the factorial of n."
N = 150
base_a = [decode.baseline_sampled(target, cp, max_new=1, temperature=1.0, seed=s)["tokens"][0] for s in range(N)]
base_b = [decode.baseline_sampled(target, cp, max_new=1, temperature=1.0, seed=s)["tokens"][0] for s in range(1000, 1000+N)]
spec   = [decode.speculative_sampled(draft, target, cp, max_new=1, gamma=GAMMA, temperature=1.0, seed=s)["tokens"][0] for s in range(N)]
tv_noise = tv(base_a, base_b)
tv_test  = tv(base_a, spec)
print(f"  TV(baseline, baseline) = {tv_noise:.3f}   (sampling-noise floor, N={N})")
print(f"  TV(baseline, spec)     = {tv_test:.3f}   (should be ~ the floor)")

# ---------- Part B: temperature sweep ----------
TEMPS = [0.2, 0.5, 0.7, 1.0]
test_prompts = prompts.load_code(60)[:24]
decode.baseline_sampled(target, "Say hello.", max_new=8)          # warmup
decode.speculative_sampled(draft, target, "Say hello.", max_new=8, gamma=GAMMA)

out = {"draft": draft, "target": target, "gamma": GAMMA, "c": c,
       "tv_noise": tv_noise, "tv_test": tv_test, "by_temp": {}}
print(f"\n== temperature sweep ({draft}->{target}, gamma={GAMMA}) ==")
for T in TEMPS:
    accs, speedups = [], []
    for i, p in enumerate(test_prompts):
        b = decode.baseline_sampled(target, p, max_new=MAX_NEW, temperature=T, seed=i)
        s = decode.speculative_sampled(draft, target, p, max_new=MAX_NEW, gamma=GAMMA, temperature=T, seed=i)
        accs.append(s["accept_rate"]); speedups.append(b["time"] / s["time"])
    a_mean, a_ci = stats.bootstrap_ci(accs)
    s_mean, s_ci = stats.bootstrap_ci(speedups)
    pred = theory.speedup(a_mean, GAMMA, c)
    out["by_temp"][T] = {"alpha": a_mean, "alpha_ci": list(a_ci),
                         "measured": s_mean, "measured_ci": list(s_ci), "predicted": pred}
    print(f"  T={T}: alpha={a_mean:.3f} [{a_ci[0]:.2f},{a_ci[1]:.2f}]  "
          f"predicted {pred:.2f}x  measured {s_mean:.2f}x [{s_ci[0]:.2f},{s_ci[1]:.2f}]")

with open("data/sampled.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote data/sampled.json")
