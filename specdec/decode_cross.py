"""Cross-tokenizer greedy speculative decoding.

When the draft and target are different model families, their tokenizers don't
share a vocabulary, so the draft's token ids mean nothing to the target. We bridge
through *text*:

  1. draft continues the committed text (in the draft's tokenizer) and proposes
     gamma tokens;
  2. decode those to a string, re-encode it with the TARGET tokenizer -> candidate
     target tokens;
  3. verify the candidates in the target's token space exactly as in same-family
     greedy speculation (accept the longest prefix the target itself would emit,
     plus one correction token).

Because verification happens entirely in target-token space, the output is still
exactly the target's greedy output — the draft only changes *speed*, never the
result. The draft keeps no KV cache (its context is re-derived from the committed
text each round, since cross-tokenizer prefixes aren't stably incremental); the
target uses the same gap-refed cache as the same-family decoder.
"""
import time
import torch
from transformers import DynamicCache
from .models import load, chat_ids, device


@torch.no_grad()
def _draft_propose(draft, dtok, prompt, gen_text, gamma, dev):
    """Greedily extend `gen_text` by gamma draft tokens; return the proposal text."""
    msgs = [{"role": "user", "content": prompt}]
    prefix = dtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = dtok(prefix + gen_text, return_tensors="pt").input_ids.to(dev)
    out = draft.generate(ids, max_new_tokens=gamma, do_sample=False,
                         pad_token_id=dtok.eos_token_id)
    new_ids = out[0, ids.shape[1]:]
    return dtok.decode(new_ids, skip_special_tokens=True)


@torch.no_grad()
def speculative_cross_greedy(draft_tag, target_tag, prompt, max_new=64, gamma=4):
    draft, dtok = load(draft_tag)
    target, ttok = load(target_tag)
    dev = device()
    eos = ttok.eos_token_id
    ids = chat_ids(ttok, prompt).to(dev)
    seq = ids[0].tolist()                 # committed sequence in TARGET tokens
    plen = len(seq)
    tcache = DynamicCache()
    stats = {"iters": 0, "draft_fwd": 0, "target_fwd": 0, "accepted": 0, "proposed": 0}

    def gap():
        return seq[tcache.get_seq_length():]

    if dev == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    done = False
    while len(seq) - plen < max_new and not done:
        stats["iters"] += 1
        gen_text = ttok.decode(seq[plen:], skip_special_tokens=True)
        proposal = _draft_propose(draft, dtok, prompt, gen_text, gamma, dev)
        stats["draft_fwd"] += gamma
        cand = ttok(proposal, add_special_tokens=False).input_ids if proposal else []

        tfeed = torch.tensor([gap() + cand], device=dev)
        out = target(input_ids=tfeed, past_key_values=tcache, use_cache=True)
        tcache = out.past_key_values
        stats["target_fwd"] += 1

        if cand:
            tlog = out.logits[0, -(len(cand) + 1):]
            tpred = tlog.argmax(-1).tolist()      # tpred[i] = target token after position i
            n_accept = 0
            for i in range(len(cand)):
                if tpred[i] == cand[i]:
                    n_accept += 1
                else:
                    break
            new_tokens = cand[:n_accept] + [tpred[n_accept]]
            stats["accepted"] += n_accept; stats["proposed"] += len(cand)
        else:
            # empty proposal -> fall back to one plain target greedy step
            new_tokens = [int(out.logits[0, -1].argmax())]
            stats["proposed"] += 0

        for k, t in enumerate(new_tokens):
            if t == eos:
                new_tokens = new_tokens[:k]; done = True; break
        if len(seq) - plen + len(new_tokens) >= max_new:
            new_tokens = new_tokens[:max_new - (len(seq) - plen)]; done = True

        seq.extend(new_tokens)
        tcache.crop(min(tcache.get_seq_length(), len(seq) - 1))

    if dev == "mps": torch.mps.synchronize()
    stats["time"] = time.perf_counter() - t0
    stats["tokens"] = seq[plen:]
    stats["accept_rate"] = stats["accepted"] / max(stats["proposed"], 1)
    stats["mean_accept_len"] = stats["accepted"] / max(stats["iters"], 1)
    return stats
