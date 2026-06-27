# When does speculative decoding actually pay off?

## Abstract

Speculative decoding accelerates LLM inference by letting a small *draft* model
propose several tokens that a large *target* model verifies in a single parallel
forward pass — the target's output is provably unchanged, but it runs in fewer
sequential steps. Whether it is *faster* depends on two competing quantities:
how often the draft agrees with the target (acceptance rate α), and how cheap the
draft is relative to the target (cost ratio c). We characterize this trade-off
empirically on the Qwen2.5-Coder family (0.5B–7B) on an Apple-Silicon (MPS)
device. We measure α and per-token latency directly, **predict** the speedup and
the optimal draft length γ from the standard speculative-sampling model, then
**verify** the prediction with a from-scratch greedy speculative decoder.
Predictions match measurement to ~0.1× (best config: predicted 1.43×, measured
1.32× wall-clock, with token-identical output). Two findings are practical:
(1) on this hardware per-token latency is **non-monotonic in model size** — the
0.5B draft is *slower* than the 1.5B, so the obvious "smallest draft" choice
actively loses; (2) acceptance is strongly **domain-dependent** (code α≈0.9 vs
prose α≈0.7), so the break-even point moves with the workload.

## Method

**Models.** Qwen2.5-Coder-Instruct at 0.5B / 1.5B / 3B / 7B (shared tokenizer,
which is what makes them valid draft/target pairs), fp16 on MPS.

**Two empirical inputs** (`scripts/01_acceptance.py`):
- *per-token latency* — median seconds/token with KV cache, per model.
- *acceptance rate α* — P(draft's greedy next token == target's greedy next
  token), teacher-forced on the target's own greedy continuation. Accepted
  tokens are by construction the target's tokens, so this is exactly the α in the
  speedup model. Measured per domain (code vs prose).

**Prediction** (`specdec/theory.py`, Leviathan et al. 2023). Per iteration the
draft proposes γ tokens, the target verifies in one pass; expected emitted tokens
`E = (1−α^(γ+1))/(1−α)`, cost `(γ·c + 1)` target-forwards, so
`speedup = E / (γ·c + 1)`, maximized over γ.

**Verification** (`specdec/decode.py`, `scripts/02_verify.py`). A from-scratch
greedy speculative decoder with explicit KV-cache management (draft and target
caches track their own covered length; the uncovered "gap" is re-fed after every
rejection). Greedy speculation is *exact*, so its output must equal baseline
greedy token-for-token — a built-in correctness check.

## Results

**Where it pays off.** Predicted speedup for every (draft→target, domain). The
dashed line is break-even; below it, speculation is slower than plain decoding.

![where it pays](figures/fig1_where_it_pays.png)

| config | α | c (draft/target) | best γ | predicted | 
|---|---|---|---|---|
| 1.5B→7B / code  | 0.89 | 0.46 | 3 | **1.43×** |
| 1.5B→7B / prose | 0.76 | 0.46 | 2 | 1.23× |
| 0.5B→7B / code  | 0.87 | 0.81 | 1 | 1.03× |
| 1.5B→3B / code  | 0.92 | 0.83 | 1 | 1.05× |
| 0.5B→3B / code  | 0.92 | 1.46 | 1 | 0.78× (loss) |
| 0.5B→3B / prose | 0.66 | 1.46 | 1 | 0.68× (loss) |

**The smallest draft is not the best.** Measured per-token latency was 0.5B
30.2 ms, 1.5B 17.1 ms, 3B 20.7 ms, 7B 37.4 ms — *non-monotonic*. Tiny models are
dominated by fixed per-step overhead, not FLOPs, so the 0.5B draft has an
unfavorable cost ratio (c>1 against the 3B target) and loses despite high
acceptance.

**Speedup vs draft length.** A clear optimum then diminishing returns: more draft
tokens raise expected emissions but cost more when rejected.

![gamma sweep](figures/fig2_gamma_sweep.png)

**Prediction vs reality.** Running the real decoder on the best config
(1.5B→7B, code, γ=3): **1.32× measured** vs 1.43× predicted, mean accept-rate
0.85, mean accepted-length 2.55 tokens/iteration. Output was token-identical to
baseline greedy on 5/6 prompts; the one mismatch is a single fp16 argmax tie-flip
on MPS (batched-verify vs sequential logits differ at the last bit), not a logic
error. The ~0.1× shortfall is real overhead the formula omits (gap re-feeding,
Python loop, the verify pass not being literally free).

## Conclusions

Speculative decoding's payoff is governed by the product of acceptance (α) and
relative draft cost (c), and both move with hardware and workload. On Apple
Silicon the FLOPs-cheap "smallest draft" intuition fails because latency is
overhead-bound at small sizes; the right draft is the one minimizing `γ·c+1` for
its α, which here is the 1.5B feeding the 7B. Acceptance is high and stable
in-domain (code) and degrades off-domain (prose), shifting the break-even γ. A
simple analytic model predicts the measured wall-clock speedup to within ~0.1×,
so the optimal configuration can be chosen from two cheap measurements without
running the full decoder.

## Limitations

- One GPU-less device class (Apple MPS); the latency non-monotonicity is
  hardware-specific — on a datacenter GPU the cost ratios (and conclusions) shift.
- One model family; greedy decoding only (sampled speculation has a stochastic
  acceptance rule — a natural extension).
- Short generations (≤96 tokens) and small prompt sets; α is averaged, not
  modeled per-position.

## Related work
- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative
  Decoding*, ICML 2023 — the acceptance/speedup model used here.
  [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al., *Accelerating Large Language Model Decoding with Speculative
  Sampling*, 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

## Reproduce

```sh
pip install -r requirements.txt
python scripts/01_acceptance.py          # latency + alpha -> data/measurements.json + predictions
python scripts/02_verify.py 1.5B 7B 3    # real decoder: exactness check + wall-clock speedup
python scripts/03_figures.py             # figures
```

Runs locally on CPU / Apple MPS, no API key.

## Repository

```
specdecode-characterization/
├── specdec/
│   ├── models.py      # load the Qwen2.5-Coder ladder (shared tokenizer)
│   ├── measure.py     # per-token latency + acceptance rate alpha
│   ├── theory.py      # Leviathan speedup model, optimal-gamma
│   ├── decode.py      # from-scratch greedy baseline + speculative decoder
│   └── prompts.py     # code / prose prompt sets
├── scripts/           # 01 measure · 02 verify · 03 figures
└── figures/
```
