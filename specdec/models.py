"""Load the Qwen2.5-Coder ladder once and reuse. Same tokenizer across the
family, which is what makes them valid draft/target pairs for speculation.
`smol-360M` is a different family (different tokenizer) for cross-tokenizer
speculation (see decode_cross.py)."""
import functools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "0.5B": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "3B":   "Qwen/Qwen2.5-Coder-3B-Instruct",
    "7B":   "Qwen/Qwen2.5-Coder-7B-Instruct",
    "smol-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def sync():
    """Block until queued GPU work is done, so timings are real (CUDA or MPS)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

@functools.lru_cache(maxsize=None)
def load(tag, dtype=None):
    name = MODELS[tag]
    dev = device()
    dtype = dtype or (torch.float16 if dev in ("mps", "cuda") else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(dev).eval()
    return model, tok

def chat_ids(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids
