#!/usr/bin/env python3
"""Multi-turn prefix-cache probe for a running llama-server.

Sends a long document as a system prefix, then asks 4 questions in one growing
conversation, and one control question on a *different* document. Reports how many
prompt tokens actually had to be evaluated vs reused from the KV cache.

Usage: prefix_probe.py <port> <label> [docfile]
"""
import json
import sys
import urllib.error
import urllib.request

port = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "run"
docfile = sys.argv[3] if len(sys.argv) > 3 else "/home/torridfish/Code/lfm/testdata/longprompt.txt"
url = f"http://127.0.0.1:{port}/v1/chat/completions"

doc = open(docfile).read()
QUESTIONS = [
    "In one short sentence: what is the binding constraint for token generation?",
    "Name the platform mentioned in the first sentence.",
    "What does the text say makes the machine quiet?",
    "Summarise the whole passage in one sentence.",
]


def ask(messages, tag):
    body = json.dumps({"max_tokens": 24, "temperature": 0, "messages": messages}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:120].replace("|", "/").replace("\n", " ")
        print(f"| {label} | {tag:22s} | {'-':>6} | {'-':>6} | {'-':>8} | {'-':>6} | HTTP {e.code}: {detail} |", flush=True)
        raise SystemExit(0)
    t = d.get("timings") or {}
    prompt_n = t.get("prompt_n")
    cache_n = t.get("cache_n")
    pp = t.get("prompt_per_second") or 0.0
    ms = t.get("prompt_ms") or 0.0
    answer = (d["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")[:60]
    print(
        f"| {label} | {tag:22s} | {prompt_n!s:>6} | {cache_n!s:>6} | {pp:8.1f} | {ms/1000:6.2f} | {answer} |",
        flush=True,
    )
    return d["choices"][0]["message"]["content"]


msgs = [{"role": "system", "content": doc}]
for i, q in enumerate(QUESTIONS, 1):
    msgs.append({"role": "user", "content": q})
    a = ask(msgs, f"turn {i} (same prefix)")
    msgs.append({"role": "assistant", "content": a})

# control: identical size, different prefix -> the cache must be rebuilt from scratch
ctrl = [{"role": "system", "content": "CONTROL " + doc}]
ctrl.append({"role": "user", "content": QUESTIONS[0]})
ask(ctrl, "control (new prefix)")
