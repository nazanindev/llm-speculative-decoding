"""Two clean, cache-free measurements that drive the whole characterization:

  alpha(draft, target) -- acceptance rate: P(draft's greedy next token ==
      target's greedy next token), conditioned on the true target prefix.
      This is exactly the quantity in the speculative-sampling speedup formula:
      accepted draft tokens are *by construction* the target's tokens, so
      teacher-forcing the draft on the target's greedy continuation and asking
      how often it agrees IS the per-step acceptance rate.

  token_latency(model) -- median seconds per generated token with KV cache.
"""
import time
import torch
from .models import load, chat_ids, device, sync


@torch.no_grad()
def target_continuation(target_tag, prompt, max_new=128):
    """Greedy continuation from the target. Returns (prompt_len, full_ids)."""
    model, tok = load(target_tag)
    ids = chat_ids(tok, prompt).to(device())
    plen = ids.shape[1]
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return plen, out  # full_ids: [1, plen + gen]


@torch.no_grad()
def alpha_on_sequence(draft_tag, plen, full_ids):
    """Fraction of continuation positions where the draft's greedy next-token
    matches the target's actual next token (teacher-forced on the true prefix)."""
    model, tok = load(draft_tag)
    full_ids = full_ids.to(device())
    logits = model(full_ids).logits  # [1, T, V]
    # position i predicts token i+1; we judge the continuation region only
    pred = logits[0, plen - 1:-1].argmax(-1)        # draft's greedy predictions
    truth = full_ids[0, plen:]                       # target's actual tokens
    n = min(len(pred), len(truth))
    if n == 0:
        return float("nan"), 0
    return float((pred[:n] == truth[:n]).float().mean()), n


@torch.no_grad()
def token_latency(tag, prompt="Write a Python function to merge two sorted lists.",
                  max_new=64, warmup=8, trials=5):
    """Per-token latency (greedy, KV cache on). Returns the median plus the raw
    per-trial samples so callers can report spread."""
    model, tok = load(tag)
    ids = chat_ids(tok, prompt).to(device())
    # warmup (kernel compile / cache alloc)
    model.generate(ids, max_new_tokens=warmup, do_sample=False,
                   pad_token_id=tok.eos_token_id)
    samples = []
    for _ in range(trials):
        sync()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        sync()
        gen = out.shape[1] - ids.shape[1]
        samples.append((time.perf_counter() - t0) / max(gen, 1))
    samples.sort()
    return samples[len(samples) // 2], samples
