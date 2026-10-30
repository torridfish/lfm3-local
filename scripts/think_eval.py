#!/usr/bin/env python3
"""A/B: does LFM3-1.7B's thinking mode actually score better?

Runs a small deterministic battery against two llama-server ports (same model, one with
thinking on and one with it off) and prints accuracy + latency + token cost.

Usage: think_eval.py <port_thinking_off> <port_thinking_on>
"""
import json
import re
import sys
import time
import urllib.request

PORTS = {"off": sys.argv[1], "on": sys.argv[2]} if len(sys.argv) > 2 else {"off": "8111", "on": "8110"}

WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]

# (name, prompt, check(reply_dict) -> bool, tools)
CASES = [
    ("math: change",
     "A shop sells pens at $3 each. I buy 7 pens and pay with a $50 note. How much change do I get? "
     "End your answer with the amount in the form $N.",
     lambda r: bool(re.search(r"\$\s*29\b", r["content"] or "")), None),
    ("math: arrival time",
     "A train leaves at 14:45 and the journey takes 3 hours 50 minutes. What time does it arrive? "
     "Answer with a time in HH:MM form.",
     lambda r: bool(re.search(r"\b18[:.]35\b", r["content"] or "")), None),
    ("math: ages",
     "Tom is twice as old as Anna. Anna is 3 years younger than Ben. Ben is 11 years old. "
     "How old is Tom? End with the number alone.",
     lambda r: bool(re.search(r"(^|\D)16(\D|$)", (r["content"] or "").split("\n")[-2:][0] or r["content"] or "")), None),
    ("logic: contrapositive",
     "If it rains, the ground is wet. The ground is not wet. Did it rain? "
     "Answer 'yes' or 'no' as the first word.",
     lambda r: (r["content"] or "").strip().lower().startswith("no"), None),
    ("json: invoice",
     "Return JSON only, with keys name, date, amount: 'Invoice from Acme Ltd dated 2026-03-04 for NT$1,250.'",
     lambda r: (lambda d: d.get("name", "").startswith("Acme") and "2026-03-04" in str(d.get("date"))
                and "1,250" in str(d.get("amount")) or "1250" in str(d.get("amount")))(_maybe_json(r["content"])), None),
    ("json: order",
     "Return JSON only, keys: order_id, date, total. 'Order A-77 placed on 2026-02-10 for 3 units at $4.50 each.'",
     lambda r: (lambda d: "A-77" in str(d.get("order_id")) and "2026-02-10" in str(d.get("date"))
                and "13.5" in str(d.get("total")))(_maybe_json(r["content"])), None),
    ("format: exactly 3 words",
     "Reply with exactly three words, all lowercase, no punctuation, naming three colours. Nothing else.",
     lambda r: bool(re.fullmatch(r"[a-z]+ [a-z]+ [a-z]+", (r["content"] or "").strip().lower())), None),
    ("tool call",
     "What is the weather in Taipei right now?",
     lambda r: bool(r["tools"]) and r["tools"][0]["function"]["name"] == "get_weather"
     and "taipei" in r["tools"][0]["function"]["arguments"].lower(), WEATHER_TOOL),
]


def _maybe_json(text):
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def run(port, prompt, tools):
    body = {"model": "lfm3", "temperature": 0, "max_tokens": 900, "messages": [{"role": "user", "content": prompt}]}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    wall = time.time() - t0
    m = d["choices"][0]["message"]
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or "",
        "tools": m.get("tool_calls") or [],
        "tokens": d["usage"]["completion_tokens"],
        "wall": wall,
        "tps": d["timings"]["predicted_per_second"],
    }


print("| case | thinking OFF | thinking ON | tokens off/on | seconds off/on |")
print("|---|---|---|---|---|")
score = {"off": 0, "on": 0}
tot_tok = {"off": 0, "on": 0}
tot_wall = {"off": 0.0, "on": 0.0}
for name, prompt, check, tools in CASES:
    res, ok = {}, {}
    for tag in ("off", "on"):
        try:
            res[tag] = run(PORTS[tag], prompt, tools)
            ok[tag] = bool(check(res[tag]))
        except Exception as e:
            res[tag] = {"content": f"ERROR {e}", "reasoning": "", "tools": [], "tokens": 0, "wall": 0, "tps": 0}
            ok[tag] = False
        score[tag] += ok[tag]
        tot_tok[tag] += res[tag]["tokens"]
        tot_wall[tag] += res[tag]["wall"]
    print(f"| {name} | {'PASS' if ok['off'] else 'fail'} | {'PASS' if ok['on'] else 'fail'} | "
          f"{res['off']['tokens']}/{res['on']['tokens']} | {res['off']['wall']:.1f}/{res['on']['wall']:.1f} |")

n = len(CASES)
print()
print(f"**score: off {score['off']}/{n} · on {score['on']}/{n}**  |  "
      f"tokens: off {tot_tok['off']} vs on {tot_tok['on']} ({tot_tok['on']/max(1,tot_tok['off']):.1f}x)  |  "
      f"wall: off {tot_wall['off']:.0f}s vs on {tot_wall['on']:.0f}s ({tot_wall['on']/max(0.1,tot_wall['off']):.1f}x)")
