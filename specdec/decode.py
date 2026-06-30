"""From-scratch decoders with KV cache.

  greedy:  baseline_greedy / speculative_greedy -- exact, output token-for-token
           equals baseline greedy (built-in correctness check).
  sampled: baseline_sampled / speculative_sampled -- accept draft token x with
           probability min(1, p(x)/q(x)); on rejection sample the correction from
           the normalized residual max(0, p-q). This makes the speculative output
           distributed *identically* to sampling from the target alone
           (Leviathan/Chen), so correctness is checked distributionally.

Caches track their own covered length and we always feed the uncovered "gap", so
draft/target stay consistent across rejections without fragile index math.
"""
import time
import torch
from transformers import DynamicCache
from .models import load, chat_ids, device


def _probs(logits_row, temperature):
    """Softmax with temperature, on CPU (stable + MPS-generator-free sampling)."""
    return torch.softmax(logits_row.float().cpu() / temperature, dim=-1)


def _sample(prob_vec, gen):
    return int(torch.multinomial(prob_vec, 1, generator=gen).item())


@torch.no_grad()
def baseline_greedy(target_tag, prompt, max_new=128):
    model, tok = load(target_tag)
    dev = device()
    ids = chat_ids(tok, prompt).to(dev)
    cache = DynamicCache()
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    cur = int(out.logits[0, -1].argmax())
    gen = [cur]
    fwd = 1
    for _ in range(max_new - 1):
        if cur == tok.eos_token_id: break
        out = model(input_ids=torch.tensor([[cur]], device=dev),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur = int(out.logits[0, -1].argmax()); fwd += 1
        gen.append(cur)
    if dev == "mps": torch.mps.synchronize()
    dt = time.perf_counter() - t0
    if gen and gen[-1] == tok.eos_token_id: gen = gen[:-1]
    return {"tokens": gen, "time": dt, "target_fwd": fwd}


@torch.no_grad()
def speculative_greedy(draft_tag, target_tag, prompt, max_new=128, gamma=4):
    draft, tok = load(draft_tag)
    target, _ = load(target_tag)
    dev = device()
    eos = tok.eos_token_id
    ids = chat_ids(tok, prompt).to(dev)
    seq = ids[0].tolist()              # full committed sequence (ids)
    plen = len(seq)
    dcache, tcache = DynamicCache(), DynamicCache()

    stats = {"iters": 0, "draft_fwd": 0, "target_fwd": 0, "accepted": 0, "proposed": 0}
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()

    def gap(cache):
        return seq[cache.get_seq_length():]

    done = False
    while len(seq) - plen < max_new and not done:
        stats["iters"] += 1
        # --- draft proposes gamma tokens (feeding its uncovered gap first) ---
        proposals = []
        feed = torch.tensor([gap(dcache)], device=dev)
        for j in range(gamma):
            out = draft(input_ids=feed, past_key_values=dcache, use_cache=True)
            dcache = out.past_key_values
            q = int(out.logits[0, -1].argmax())
            proposals.append(q)
            feed = torch.tensor([[q]], device=dev)
            stats["draft_fwd"] += 1
        # note: dcache now covers seq + proposals[:-1] (last proposal not fed)

        # --- target verifies all proposals in one pass ---
        tfeed = torch.tensor([gap(tcache) + proposals], device=dev)
        out = target(input_ids=tfeed, past_key_values=tcache, use_cache=True)
        tcache = out.past_key_values
        stats["target_fwd"] += 1
        # last gamma+1 logits: predictions after [last_committed, q1..q_gamma]
        tlog = out.logits[0, -(gamma + 1):]
        tpred = tlog.argmax(-1).tolist()      # tpred[i] = target's token after position i

        # --- accept longest matching prefix, then one correction/bonus token ---
        n_accept = 0
        for i in range(gamma):
            if tpred[i] == proposals[i]:
                n_accept += 1
            else:
                break
        new_tokens = proposals[:n_accept] + [tpred[n_accept]]   # +correction (or bonus if n==gamma)
        stats["accepted"] += n_accept
        stats["proposed"] += gamma

        # truncate at eos
        for k, t in enumerate(new_tokens):
            if t == eos:
                new_tokens = new_tokens[:k]; done = True; break
        if len(seq) - plen + len(new_tokens) >= max_new:
            new_tokens = new_tokens[:max_new - (len(seq) - plen)]; done = True

        seq.extend(new_tokens)
        keep = len(seq) - 1                    # caches hold everything except the new last token
        tcache.crop(min(tcache.get_seq_length(), keep))
        dcache.crop(min(dcache.get_seq_length(), keep))

    if dev == "mps": torch.mps.synchronize()
    stats["time"] = time.perf_counter() - t0
    gen = seq[plen:]
    stats["tokens"] = gen
    stats["accept_rate"] = stats["accepted"] / max(stats["proposed"], 1)
    stats["mean_accept_len"] = stats["accepted"] / max(stats["iters"], 1)
    return stats


@torch.no_grad()
def baseline_sampled(target_tag, prompt, max_new=128, temperature=1.0, seed=0):
    model, tok = load(target_tag)
    dev = device()
    gen_rng = torch.Generator().manual_seed(seed)
    ids = chat_ids(tok, prompt).to(dev)
    cache = DynamicCache()
    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    cur = _sample(_probs(out.logits[0, -1], temperature), gen_rng)
    gen = [cur]; fwd = 1
    for _ in range(max_new - 1):
        if cur == tok.eos_token_id: break
        out = model(input_ids=torch.tensor([[cur]], device=dev),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        cur = _sample(_probs(out.logits[0, -1], temperature), gen_rng); fwd += 1
        gen.append(cur)
    if dev == "mps": torch.mps.synchronize()
    if gen and gen[-1] == tok.eos_token_id: gen = gen[:-1]
    return {"tokens": gen, "time": time.perf_counter() - t0, "target_fwd": fwd}


@torch.no_grad()
def speculative_sampled(draft_tag, target_tag, prompt, max_new=128, gamma=4,
                        temperature=1.0, seed=0):
    draft, tok = load(draft_tag)
    target, _ = load(target_tag)
    dev = device()
    eos = tok.eos_token_id
    rng = torch.Generator().manual_seed(seed)
    ids = chat_ids(tok, prompt).to(dev)
    seq = ids[0].tolist()
    plen = len(seq)
    dcache, tcache = DynamicCache(), DynamicCache()
    stats = {"iters": 0, "draft_fwd": 0, "target_fwd": 0, "accepted": 0, "proposed": 0}

    def gap(cache):
        return seq[cache.get_seq_length():]

    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    done = False
    while len(seq) - plen < max_new and not done:
        stats["iters"] += 1
        # --- draft samples gamma tokens, recording its full distribution q ---
        proposals, qdists = [], []
        feed = torch.tensor([gap(dcache)], device=dev)
        for _ in range(gamma):
            out = draft(input_ids=feed, past_key_values=dcache, use_cache=True)
            dcache = out.past_key_values
            q = _probs(out.logits[0, -1], temperature)
            x = _sample(q, rng)
            proposals.append(x); qdists.append(q)
            feed = torch.tensor([[x]], device=dev)
            stats["draft_fwd"] += 1

        # --- target distributions p over the same positions, one pass ---
        tfeed = torch.tensor([gap(tcache) + proposals], device=dev)
        out = target(input_ids=tfeed, past_key_values=tcache, use_cache=True)
        tcache = out.past_key_values
        stats["target_fwd"] += 1
        tlog = out.logits[0, -(gamma + 1):]
        pdists = [_probs(tlog[i], temperature) for i in range(gamma + 1)]
        # models in a family can pad their vocab differently (e.g. 7B 152064 vs
        # 1.5B 151936); align p and q on the shared tokenizer vocab.
        V = min(pdists[0].shape[0], qdists[0].shape[0])
        pdists = [(d[:V] / d[:V].sum()) for d in pdists]
        qdists = [(d[:V] / d[:V].sum()) for d in qdists]

        # --- probabilistic accept / residual-correction ---
        n_accept = 0
        correction = None
        for i in range(gamma):
            x = proposals[i]
            ratio = (pdists[i][x] / qdists[i][x]).item()
            if torch.rand(1, generator=rng).item() < min(1.0, ratio):
                n_accept += 1
            else:
                resid = torch.clamp(pdists[i] - qdists[i], min=0)
                s = resid.sum()
                resid = resid / s if s > 0 else pdists[i]
                correction = _sample(resid, rng)
                break
        if correction is None:                       # all accepted -> bonus from p_{gamma+1}
            correction = _sample(pdists[gamma], rng)
        new_tokens = proposals[:n_accept] + [correction]
        stats["accepted"] += n_accept; stats["proposed"] += gamma

        for k, t in enumerate(new_tokens):
            if t == eos:
                new_tokens = new_tokens[:k]; done = True; break
        if len(seq) - plen + len(new_tokens) >= max_new:
            new_tokens = new_tokens[:max_new - (len(seq) - plen)]; done = True

        seq.extend(new_tokens)
        keep = len(seq) - 1
        tcache.crop(min(tcache.get_seq_length(), keep))
        dcache.crop(min(dcache.get_seq_length(), keep))

    if dev == "mps": torch.mps.synchronize()
    stats["time"] = time.perf_counter() - t0
    stats["tokens"] = seq[plen:]
    stats["accept_rate"] = stats["accepted"] / max(stats["proposed"], 1)
    stats["mean_accept_len"] = stats["accepted"] / max(stats["iters"], 1)
    return stats
