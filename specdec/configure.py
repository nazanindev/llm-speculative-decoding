"""Deploy-decision tool. Given measured acceptance + latencies, answer the
question an inference service actually asks: should I enable speculative decoding
for this draft/target pair and workload, and with what gamma?

Reads data/measurements.json (alpha, per-model latency) and, if present,
data/serving.json (batch roofline). Returns a recommendation dict.
"""
import os, json
from . import theory


def _load(name):
    path = os.path.join("data", name)
    return json.load(open(path)) if os.path.exists(path) else None


def recommend(draft, target, domain="code", batch=1, gamma_max=8):
    M = _load("measurements.json")
    if M is None:
        raise FileNotFoundError("run scripts/01_acceptance.py first")
    key = f"{draft}->{target}/{domain}"
    if key not in M["alpha"]:
        raise KeyError(f"no measurement for {key}; available: {list(M['alpha'])}")
    alpha = M["alpha"][key]
    c = M["latency"][draft] / M["latency"][target]

    # batch>1: use the serving roofline if we measured it (verify stops being free)
    serving = _load("serving.json")
    if batch > 1 and serving and serving["draft"] == draft and serving["target"] == target:
        bb = serving["by_batch"]
        Bkey = min(bb, key=lambda b: abs(int(b) - batch))   # nearest measured batch
        s = bb[Bkey]
        E = theory.expected_tokens(alpha, serving["gamma"])
        speedup = E * s["t_target_1"] / (serving["gamma"] * s["t_draft_1"] + s["t_target_g"])
        g = serving["gamma"]
        basis = f"serving roofline at batch≈{Bkey}"
    else:
        g, speedup = theory.best_gamma(alpha, c, gamma_max)
        basis = "batch=1 latency model"

    return {
        "pair": key, "batch": batch, "alpha": round(alpha, 3), "cost_ratio": round(c, 3),
        "recommend_speculation": speedup > 1.0,
        "gamma": g, "expected_speedup": round(speedup, 2), "basis": basis,
        "reason": (f"speculate with gamma={g} for ~{speedup:.2f}x" if speedup > 1.0
                   else f"do NOT speculate (best ~{speedup:.2f}x < 1.0)"),
    }


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "1.5B"
    t = sys.argv[2] if len(sys.argv) > 2 else "7B"
    dom = sys.argv[3] if len(sys.argv) > 3 else "code"
    b = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(json.dumps(recommend(d, t, dom, b), indent=1))
