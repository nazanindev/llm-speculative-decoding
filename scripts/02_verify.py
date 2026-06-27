"""Stage 2: run the real decoder. Check that greedy speculation is EXACT
(token-for-token == baseline greedy), then measure the wall-clock speedup and
compare it to the stage-1 prediction.

Usage: python scripts/02_verify.py 1.5B 7B 3      # draft target gamma
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import decode, prompts

draft  = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
target = sys.argv[2] if len(sys.argv) > 2 else "7B"
gamma  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_NEW = 96
test_prompts = prompts.code[:6]

base_tot = spec_tot = 0.0
exact = True
acc_rates, accept_lens = [], []
print(f"draft={draft} target={target} gamma={gamma}\n")
# warmup: compile kernels / allocate caches so the first timed prompt is fair
decode.baseline_greedy(target, "Say hello.", max_new=8)
decode.speculative_greedy(draft, target, "Say hello.", max_new=8, gamma=gamma)
for p in test_prompts:
    b = decode.baseline_greedy(target, p, max_new=MAX_NEW)
    s = decode.speculative_greedy(draft, target, p, max_new=MAX_NEW, gamma=gamma)
    ok = b["tokens"] == s["tokens"]
    exact &= ok
    base_tot += b["time"]; spec_tot += s["time"]
    acc_rates.append(s["accept_rate"]); accept_lens.append(s["mean_accept_len"])
    print(f"  exact={ok}  base {b['time']:.2f}s  spec {s['time']:.2f}s  "
          f"x{b['time']/s['time']:.2f}  accept={s['accept_rate']:.2f}  "
          f"len={s['mean_accept_len']:.2f}  ({len(b['tokens'])} tok)")

print(f"\nEXACT MATCH (all prompts): {exact}")
print(f"total: baseline {base_tot:.2f}s  speculative {spec_tot:.2f}s")
print(f"measured speedup: {base_tot/spec_tot:.2f}x   "
      f"mean accept-rate {sum(acc_rates)/len(acc_rates):.3f}  "
      f"mean accept-len {sum(accept_lens)/len(accept_lens):.2f}")
