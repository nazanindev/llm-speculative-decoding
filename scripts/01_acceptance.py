"""Stage 1: the two empirical inputs to the speedup model.

  (a) per-token latency for every model
  (b) acceptance rate alpha for each (draft, target) pair, per domain

Then combine via specdec.theory to PREDICT the speedup and best gamma.
Writes data/measurements.json.

Usage: python scripts/01_acceptance.py            # default pairs + domains
"""
import os, sys, json, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import measure, theory, prompts

DRAFTS  = ["0.5B", "1.5B"]
TARGETS = ["3B", "7B"]
DOMAINS = {"code": prompts.code, "prose": prompts.prose}
MAX_NEW = 96

def main():
    out = {"latency": {}, "alpha": {}, "predicted": {}}

    print("== per-token latency (s/token) ==")
    for m in ["0.5B", "1.5B", "3B", "7B"]:
        lat = measure.token_latency(m)
        out["latency"][m] = lat
        print(f"  {m:>5}: {lat*1000:7.1f} ms/token")

    print("\n== acceptance rate alpha ==")
    for target in TARGETS:
        for domain, plist in DOMAINS.items():
            # one target continuation per prompt, reused across all drafts
            conts = [measure.target_continuation(target, p, max_new=MAX_NEW) for p in plist]
            for draft in DRAFTS:
                if draft == target:
                    continue
                vals, ntok = [], 0
                for plen, full in conts:
                    a, n = measure.alpha_on_sequence(draft, plen, full)
                    if n > 0:
                        vals.append(a); ntok += n
                alpha = statistics.mean(vals)
                key = f"{draft}->{target}/{domain}"
                c = out["latency"][draft] / out["latency"][target]
                g, s = theory.best_gamma(alpha, c)
                out["alpha"][key] = alpha
                out["predicted"][key] = {"c": c, "best_gamma": g, "speedup": s}
                print(f"  {key:>22}: alpha={alpha:.3f}  c={c:.3f}  "
                      f"-> best gamma={g}, predicted speedup={s:.2f}x")

    os.makedirs("data", exist_ok=True)
    with open("data/measurements.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote data/measurements.json")

if __name__ == "__main__":
    main()
