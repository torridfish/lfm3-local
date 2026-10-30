#!/usr/bin/env python3
"""Concurrency soak test for a running llama-server.

Hits the server with N requests from `workers` threads, reports throughput, latency
percentiles and failures. Used to check that LFM3 (which inherits LFM2's short-conv
layers, llama.cpp #14670) survives concurrent slots, and to measure how much of the
decode bandwidth actually parallelises.

Usage: soak.py <port> [requests] [workers] [max_tokens]
"""
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

port = sys.argv[1] if len(sys.argv) > 1 else "8130"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 32
url = f"http://127.0.0.1:{port}/v1/chat/completions"

lock = threading.Lock()
errors = []


def one(i):
    body = json.dumps({
        "max_tokens": max_tokens, "temperature": 0.2,
        "messages": [{"role": "user", "content": f"Request {i}: reply with one short sentence about rain."}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
            code = r.status
        el = time.time() - t0
        content = (d["choices"][0]["message"].get("content") or "").strip()
        toks = d.get("usage", {}).get("completion_tokens", 0)
        return {"i": i, "code": code, "el": el, "tokens": toks if content else -1}
    except Exception as e:
        el = time.time() - t0
        with lock:
            errors.append(f"#{i}: {type(e).__name__}: {e}")
        return {"i": i, "code": 0, "el": el, "tokens": -1}


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * q))))
    return s[k]


t_start = time.time()
with ThreadPoolExecutor(max_workers=workers) as ex:
    results = list(ex.map(one, range(n)))
wall = time.time() - t_start

lat = [r["el"] for r in results]
ok = [r for r in results if r["code"] == 200 and r["tokens"] > 0]
toks = sum(r["tokens"] for r in ok)
codes = {}
for r in results:
    codes[r["code"]] = codes.get(r["code"], 0) + 1

print(f"requests: {len(results)}  workers: {workers}  wall: {wall:.1f}s")
print(f"throughput: {len(results)/wall:.2f} req/s  {toks/wall:.1f} completion-tokens/s")
print(f"status: {codes}   failed/empty: {len(results) - len(ok)}")
print(f"latency: mean {statistics.mean(lat):.2f}s  p50 {pct(lat,.50):.2f}s  "
      f"p90 {pct(lat,.90):.2f}s  p99 {pct(lat,.99):.2f}s  max {max(lat):.2f}s")
if ok:
    print(f"tokens per request: mean {toks/len(ok):.1f}")
if errors:
    print("errors:")
    for e in errors[:5]:
        print("  " + e)
    if len(errors) > 5:
        print(f"  ... {len(errors)-5} more")
