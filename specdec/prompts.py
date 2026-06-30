"""Prompt sets.

Real benchmarks (cached to data/ on first use, then offline + reproducible):
  load_code(n)  -- first n HumanEval problems, framed as a completion instruction.
  load_prose(n) -- first n Dolly open_qa instructions (no context) as off-domain.

The short hand-written `code` / `prose` lists below are kept as a zero-download
quickstart for the demo scripts.
"""
import os, json

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _cache(name, build):
    os.makedirs(_DATA, exist_ok=True)
    path = os.path.join(_DATA, name)
    if os.path.exists(path):
        return json.load(open(path))
    items = build()
    json.dump(items, open(path, "w"), indent=1)
    return items


def load_code(n=40):
    def build():
        import datasets
        d = datasets.load_dataset("openai_humaneval", split="test")
        return ["Complete this Python function. Return only the function.\n\n" + r["prompt"].strip()
                for r in d.select(range(min(n, len(d))))]
    return _cache(f"prompts_code_{n}.json", build)


def load_prose(n=40):
    def build():
        import datasets
        d = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
        out = []
        for r in d:
            if r["category"] == "open_qa" and not r["context"].strip():
                out.append(r["instruction"].strip())
            if len(out) >= n:
                break
        return out
    return _cache(f"prompts_prose_{n}.json", build)


code = [
    "Write a Python function to merge two sorted lists into one sorted list.",
    "Implement binary search over a sorted list, returning the index or -1.",
    "Write a function that returns the nth Fibonacci number iteratively.",
    "Implement quicksort in Python.",
    "Write a function to check whether a string is a palindrome.",
    "Implement a function to compute the greatest common divisor of two ints.",
    "Write a Python function that flattens a nested list of arbitrary depth.",
    "Implement a function to count word frequencies in a string, returning a dict.",
    "Write a function to transpose a matrix represented as a list of lists.",
    "Implement a function to find all prime numbers up to n using a sieve.",
    "Write a function that removes duplicates from a list while preserving order.",
    "Implement a function to reverse a singly linked list.",
]

prose = [
    "Explain what causes the seasons on Earth.",
    "Describe the difference between weather and climate.",
    "Summarize how a bill becomes a law in the United States.",
    "Explain why the sky appears blue during the day.",
    "Describe the water cycle in a few sentences.",
    "Explain the concept of supply and demand in economics.",
    "Describe how vaccines train the immune system.",
    "Explain what black holes are and how they form.",
]
