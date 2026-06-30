"""Stage 7: the serving dimension. Speculative decoding is usually evaluated at
batch=1 (one request, latency-bound), where the target's verify of gamma+1 tokens
is nearly free. A real inference server batches requests for throughput, which
drives the GPU compute-bound and makes the verify cost real. We measure the
roofline -- target forward latency vs batch at chunk=1 (a decode step) and
chunk=gamma+1 (a verify step) -- and turn it into the batched speedup:

    speedup(B) = E * T_target(B,1) / ( gamma * T_draft(B,1) + T_target(B,gamma+1) )

with E = (1 - a^(gamma+1))/(1 - a) tokens emitted per iteration. At small B the
verify is free and speedup ~ E; as B grows the verify costs full price and the
speedup decays toward (and past) break-even.

Usage: /usr/bin/python3 scripts/07_serving.py 1.5B 7B
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specdec import measure, theory

draft  = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
target = sys.argv[2] if len(sys.argv) > 2 else "7B"
GAMMA = 3
# go high enough to drive the GPU compute-bound: the verify pass moves B*(gamma+1)
# tokens, and the memory->compute transition is around ~200 tokens/pass, so the
# rolloff only appears once batch * (gamma+1) clears that.
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

M = json.load(open("data/measurements.json"))
alpha = M["alpha"][f"{draft}->{target}/code"]
E = theory.expected_tokens(alpha, GAMMA)

out = {"draft": draft, "target": target, "gamma": GAMMA, "alpha": alpha,
       "expected_tokens": E, "by_batch": {}}
print(f"draft={draft} target={target} gamma={GAMMA} alpha={alpha:.3f} E={E:.2f} tok/iter\n")
print(f"{'B':>3} | {'T_tgt(1)':>9} | {'T_tgt(g+1)':>11} | {'verify x':>8} | "
      f"{'T_drf(1)':>9} | {'speedup':>7}")
print("-" * 60)
import torch
for B in BATCHES:
    try:
        t1 = measure.forward_latency(target, batch=B, chunk=1)
        tg = measure.forward_latency(target, batch=B, chunk=GAMMA + 1)
        td = measure.forward_latency(draft, batch=B, chunk=1)
    except RuntimeError as e:                 # out of memory at large batch -> stop
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"{B:>3} | out of memory ({str(e)[:40]}...) -- stopping batch sweep")
        break
    verify_factor = tg / t1
    speedup = E * t1 / (GAMMA * td + tg)
    out["by_batch"][B] = {"t_target_1": t1, "t_target_g": tg, "t_draft_1": td,
                          "verify_factor": verify_factor, "speedup": speedup}
    print(f"{B:>3} | {t1*1000:8.1f} | {tg*1000:10.1f} | {verify_factor:7.2f} | "
          f"{td*1000:8.1f} | {speedup:6.2f}x")

# crossover: first batch where speedup drops below 1
below = [B for B in out["by_batch"] if out["by_batch"][B]["speedup"] < 1.0]
out["crossover_batch"] = below[0] if below else None
print(f"\ncrossover (speedup<1) at batch: {out['crossover_batch']}")

with open("data/serving.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote data/serving.json")
