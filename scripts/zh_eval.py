#!/usr/bin/env python3
"""Traditional-Chinese knowledge probe on TMMLU+ (ikala/tmmluplus).

Scores multiple-choice questions by the probability the model assigns to the answer
letter, read from one forward pass (`max_tokens=1, logprobs, top_logprobs`). That is
~20x cheaper than generating and parsing, and avoids format compliance problems in
small models.

Usage:
    zh_eval.py --port 8120 --label "LFM3-1.7B Q4" --subjects computer_science,physics
    zh_eval.py --port 8121 --label "Qwen3.5-2B Q4" --n 50 --workers 4

The CSV cache lives in $LFM_TMMLU_DIR (default ~/models/tmmluplus).
"""
import argparse
import csv
import json
import math
import os
import pathlib
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_SUBJECTS = [
    "computer_science", "physics", "junior_chemistry", "junior_math_exam", "junior_science_exam",
    "junior_social_studies", "introduction_to_law", "economics", "geography_of_taiwan",
    "traditional_chinese_medicine_clinical_medicine",
]
BASE = "https://huggingface.co/datasets/ikala/tmmluplus/resolve/main/data"
CACHE = pathlib.Path(os.environ.get("LFM_TMMLU_DIR", pathlib.Path.home() / "models/tmmluplus"))
LETTERS = ["A", "B", "C", "D"]

PROMPT = (
    "以下是一道選擇題。請只回答正確選項的字母（A、B、C 或 D），不要解釋。\n\n"
    "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
)


def fetch_csv(subject):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{subject}_test.csv"
    if not path.exists():
        url = f"{BASE}/{subject}_test.csv"
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def load_questions(subjects, n, seed=1234):
    rng = random.Random(seed)
    out = []
    for subj in subjects:
        try:
            path = fetch_csv(subj)
        except Exception as e:
            print(f"skip subject {subj}: {e}", file=sys.stderr)
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        rng.shuffle(rows)
        for r in rows[:n]:
            if r["answer"] not in LETTERS:
                continue
            out.append({
                "subject": subj,
                "prompt": PROMPT.format(q=r["question"].strip(), a=r["A"].strip(),
                                        b=r["B"].strip(), c=r["C"].strip(), d=r["D"].strip()),
                "answer": r["answer"],
            })
    return out


def ask_generate(port, prompt, model="local"):
    """Generation-based scoring: let the model emit the letter, parse it."""
    body = {
        "model": model, "max_tokens": 8, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    text = (d["choices"][0]["message"]["content"] or "").strip().upper()
    for ch in text:
        if ch in LETTERS:
            return ch
    return None


def ask(port, prompt, model="local"):
    body = {
        "model": model, "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": 10,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    lp = d["choices"][0]["logprobs"]
    if not lp or not lp.get("content"):
        return None
    scores = {}
    for tok in lp["content"][0]["top_logprobs"]:
        t = tok["token"].strip().upper()
        if t in LETTERS and t not in scores:
            scores[t] = tok["logprob"]
    if not scores:
        return None
    floor = min(scores.values()) - 12.0          # unassigned letters: heavily penalised
    total = sum(math.exp(scores.get(L, floor)) for L in LETTERS)
    probs = {L: math.exp(scores.get(L, floor)) / total for L in LETTERS}
    return max(probs, key=probs.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    ap.add_argument("--n", type=int, default=40, help="questions per subject")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="local")
    ap.add_argument("--mode", choices=["logprob", "generate"], default="logprob",
                    help="logprob = probability of the answer letter (default); generate = emit and parse")
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    qs = load_questions(subjects, args.n)

    asker = ask_generate if args.mode == "generate" else ask

    def work(q):
        try:
            pred = asker(args.port, q["prompt"], args.model)
        except Exception:
            pred = None
        return q, pred

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(work, qs))

    per_sub = {}
    for q, pred in results:
        s = per_sub.setdefault(q["subject"], [0, 0])
        s[1] += 1
        if pred == q["answer"]:
            s[0] += 1

    print(f"### {args.label}  (mode={args.mode})")
    print()
    print("| subject | acc | n |")
    print("|---|---:|---:|")
    tot_c = tot_n = 0
    for subj in subjects:
        if subj not in per_sub:
            continue
        c, n = per_sub[subj]
        tot_c += c
        tot_n += n
        print(f"| {subj} | {c/n*100:.1f}% | {n} |")
    print(f"| **overall** | **{tot_c/max(1,tot_n)*100:.1f}%** | {tot_n} |")
    print()
    print(f"raw: {tot_c}/{tot_n}")
    return tot_c, tot_n


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
