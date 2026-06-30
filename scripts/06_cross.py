"""Stage 6: cross-tokenizer speculation. A non-Qwen draft (SmolLM2-360M) drafts
for the Qwen-7B target through the decode->re-encode text bridge. We check the
output is still exactly target-greedy, then compare acceptance + speedup against
the same-family 1.5B->7B draft. Writes data/cross.json.

Usage: /usr/bin/python3 scripts/06_cross.py        # downloads SmolLM2-360M once
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import decode, decode_cross, prompts, stats

TARGET = sys.argv[1] if len(sys.argv) > 1 else "7B"
SAME_DRAFT = sys.argv[2] if len(sys.argv) > 2 else "1.5B"
GAMMA = 4
MAX_NEW = 64
test_prompts = prompts.load_code(60)[:20]

# warmup (also triggers the SmolLM download on first run)
decode.baseline_greedy(TARGET, "Say hello.", max_new=8)
decode_cross.speculative_cross_greedy("smol-360M", TARGET, "Say hello.", max_new=8, gamma=GAMMA)
decode.speculative_greedy("1.5B", TARGET, "Say hello.", max_new=8, gamma=GAMMA)

out = {"target": TARGET, "gamma": GAMMA, "configs": {}}

def run(label, fn):
    speedups, accepts, exact = [], [], 0
    for p in test_prompts:
        b = decode.baseline_greedy(TARGET, p, max_new=MAX_NEW)
        s = fn(p)
        speedups.append(b["time"] / s["time"])
        accepts.append(s["accept_rate"])
        exact += int(b["tokens"] == s["tokens"])
    sp, sci = stats.bootstrap_ci(speedups)
    ac, _ = stats.bootstrap_ci(accepts)
    out["configs"][label] = {"speedup": sp, "speedup_ci": list(sci),
                             "accept_rate": ac, "exact": exact, "n": len(test_prompts)}
    print(f"  {label:>26}: accept={ac:.3f}  speedup={sp:.2f}x [{sci[0]:.2f},{sci[1]:.2f}]  "
          f"exact={exact}/{len(test_prompts)}")

print(f"cross-tokenizer vs same-family draft (target {TARGET}, gamma={GAMMA}):")
run(f"smol-360M->{TARGET} (cross-fam)",
    lambda p: decode_cross.speculative_cross_greedy("smol-360M", TARGET, p, max_new=MAX_NEW, gamma=GAMMA))
run(f"{SAME_DRAFT}->{TARGET} (same-fam)",
    lambda p: decode.speculative_greedy(SAME_DRAFT, TARGET, p, max_new=MAX_NEW, gamma=GAMMA))

with open("data/cross.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote data/cross.json")
