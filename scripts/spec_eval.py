#!/usr/bin/env python3
"""Measure speculative-decoding payoff across workload shapes.

Speculation only pays when the model's output is predictable from context (structured
JSON, code, verbatim quotes). Free-form chat is the control where it should be neutral
or negative.

Usage: spec_eval.py <port> <label>
"""
import json
import statistics
import sys
import time
import urllib.request

port = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "run"
url = f"http://127.0.0.1:{port}/v1/chat/completions"
MAXTOK = 160

DOC = (
    "Lunar Lake pairs a low-power SoC with on-package memory. The platform keeps the "
    "machine quiet and efficient, but it also leaves memory bandwidth as the binding "
    "constraint for token generation. Local inference therefore favours aggressively "
    "quantised weights, because decode speed is limited by bytes read per token rather "
    "than by arithmetic throughput. Prefill behaves the other way round: it is compute "
    "bound, so the integrated GPU wins there by a wide margin. Prefix caching changes the "
    "economics of multi-turn work on long documents, provided the sliding-window cache is "
    "kept at full size, otherwise roughly one thousand tokens must be recomputed per turn."
)

WORKLOADS = [
    ("chat (free-form)",
     "Explain in detail how memory bandwidth limits token generation in local LLMs."),
    ("json (structured)",
     "Return JSON only, with keys vendor, date, subtotal, tax, total. Extracted from: "
     "'Invoice 2291 from Acme Semiconductor Ltd, dated 2026-03-04. Three line items: "
     "2x LPDDR5X module @ NT$4,200, 1x test fixture @ NT$11,500, 4x thermal pad @ NT$180. "
     "Sales tax is 5%%. Total includes tax.'"),
    ("code (repetitive)",
     "Rewrite this function so it returns a new list without mutating the input, and "
     "return only the code:\n\n"
     "def scale(values, factor):\n"
     "    for i in range(len(values)):\n"
     "        values[i] = values[i] * factor\n"
     "    return values\n"),
    ("quote (verbatim)",
     "Quote the exact sentence from the passage that states what limits token generation, "
     "then quote the sentence about prefill. Return only the two quotes as a JSON array.\n\n"
     "PASSAGE:\n" + DOC),
]


def ask(prompt):
    body = json.dumps({
        "max_tokens": MAXTOK, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    wall = time.time() - t0
    t = d.get("timings") or {}
    return {
        "tg": t.get("predicted_per_second") or 0.0,
        "n": d["usage"]["completion_tokens"],
        "wall": wall,
    }


print(f"### {label}")
print()
print("| workload | tg (t/s) | tokens | wall (s) |")
print("|---|---:|---:|---:|")
tgs = []
for name, prompt in WORKLOADS:
    try:
        r = ask(prompt)
    except Exception as e:
        print(f"| {name} | error: {type(e).__name__} | - | - |")
        continue
    tgs.append(r["tg"])
    print(f"| {name} | {r['tg']:.1f} | {r['n']} | {r['wall']:.2f} |")
if tgs:
    print(f"| **mean** | **{statistics.mean(tgs):.1f}** | | |")
print()
