"""Speculative-decoding speedup model (Leviathan et al., 2023).

Per speculative iteration the draft proposes `gamma` tokens, the target
verifies them in one parallel forward pass. With per-step acceptance rate
`alpha`, the expected number of tokens emitted per iteration is

    E[tokens] = (1 - alpha**(gamma+1)) / (1 - alpha)

Each iteration costs `gamma` draft forwards + 1 target forward. Writing the
draft/target cost ratio as c = latency_draft / latency_target, the wall-clock
cost per iteration (in target-forward units) is (gamma*c + 1), so

    speedup = E[tokens] / (gamma*c + 1)

vs plain autoregressive target decoding (1 token / target forward).
"""


def expected_tokens(alpha, gamma):
    if alpha >= 1.0:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c):
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def best_gamma(alpha, c, gmax=16):
    """gamma that maximizes predicted speedup, and that speedup."""
    best_g, best_s = 1, 0.0
    for g in range(1, gmax + 1):
        s = speedup(alpha, g, c)
        if s > best_s:
            best_g, best_s = g, s
    return best_g, best_s
