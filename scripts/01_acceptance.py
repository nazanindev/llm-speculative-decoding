"""Stage 1: the two empirical inputs to the speedup model.

  (a) per-token latency for every model (median of trials + spread)
  (b) acceptance rate alpha for each (draft, target) pair, per domain,
      with a bootstrap 95% CI over prompts

Then combine via specdec.theory to PREDICT the speedup and best gamma.
Writes data/measurements.json.

Usage: /usr/bin/python3 scripts/01_acceptance.py
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import measure, theory, prompts, stats

DRAFTS  = ["0.5B", "1.5B"]
TARGETS = ["3B", "7B"]
DOMAINS = {"code": prompts.load_code(60), "prose": prompts.load_prose(40)}
MAX_NEW = 96

def main():
    out = {"latency": {}, "latency_samples": {}, "alpha": {}, "alpha_ci": {}, "predicted": {}}

    print("== per-token latency (s/token, median of trials) ==")
    for m in ["0.5B", "1.5B", "3B", "7B"]:
        med, samples = measure.token_latency(m)
        out["latency"][m] = med
        out["latency_samples"][m] = samples
        print(f"  {m:>5}: {med*1000:7.1f} ms/token  (min {min(samples)*1000:.1f}, max {max(samples)*1000:.1f})")

    print("\n== acceptance rate alpha (bootstrap 95% CI over prompts) ==")
    for target in TARGETS:
        for domain, plist in DOMAINS.items():
            conts = [measure.target_continuation(target, p, max_new=MAX_NEW) for p in plist]
            for draft in DRAFTS:
                if draft == target:
                    continue
                per_prompt = []
                for plen, full in conts:
                    a, n = measure.alpha_on_sequence(draft, plen, full)
                    if n > 0:
                        per_prompt.append(a)
                alpha, (lo, hi) = stats.bootstrap_ci(per_prompt)
                key = f"{draft}->{target}/{domain}"
                c = out["latency"][draft] / out["latency"][target]
                g, s = theory.best_gamma(alpha, c)
                out["alpha"][key] = alpha
                out["alpha_ci"][key] = [lo, hi]
                out["predicted"][key] = {"c": c, "best_gamma": g, "speedup": s}
                print(f"  {key:>22}: alpha={alpha:.3f} [{lo:.2f},{hi:.2f}]  c={c:.3f}  "
                      f"-> best gamma={g}, predicted speedup={s:.2f}x")

    os.makedirs("data", exist_ok=True)
    with open("data/measurements.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote data/measurements.json")

if __name__ == "__main__":
    main()
