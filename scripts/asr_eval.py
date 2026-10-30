#!/usr/bin/env python3
"""Round-trip ASR test for LFM3-Omni-2B.

No human speech samples are available, so each phrase is first synthesised by the same
model's TTS head and then transcribed back by its ASR path. That measures internal
consistency, not real-world microphone accuracy -- synthetic speech is cleaner than a room.
Treat the numbers as an upper bound.

Usage: asr_eval.py [--keep]
"""
import os
import re
import subprocess
import sys
import tempfile
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUNTIME = os.path.join(ROOT, "runtimes", "ubuntu-vulkan-x64")
MODEL = os.path.expanduser("~/models/lfm3/LFM3-Omni-2B-GGUF/LFM3-Omni-2B-Q4_K_M.gguf")
MMPROJ = os.path.expanduser("~/models/lfm3/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-Q4_K_M.gguf")

PHRASES = [
    ("plain", "Hello, this is a local text to speech test running fully offline."),
    ("numbers", "The meeting is on March fourth at two thirty PM in room twelve."),
    ("tech", "The Arc 130V graphics uses LPDDR5X memory at eighty five hundred "
             "megatransfers per second."),
    ("homophone", "Please read the red book before you write the review."),
    ("long", "Quantising weights to four bits reduces the bytes read per token, which "
             "matters because token generation on a laptop is limited by memory "
             "bandwidth rather than by arithmetic throughput."),
    ("tw-place", "Taichung is about an hour south of Taipei by high speed rail."),
]


_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1000}
_NUMWORDS = set(_UNITS) | set(_TENS) | set(_SCALE)


def _digits(tokens):
    """Turn spoken numbers into digits the way a transcriber would write them.

    Without this, 'March fourth' vs 'March 4th', or 'eighty five hundred' vs '8500',
    is scored as an error even though the transcription is right.
    """
    out, run = [], []

    def flush():
        if not run:
            return
        if any(t in _SCALE for t in run):
            total, acc = 0, 0
            for t in run:
                if t in _SCALE:
                    acc = (acc or 1) * _SCALE[t]
                    if t == "thousand":
                        total += acc
                        acc = 0
                else:
                    acc += _UNITS.get(t, _TENS.get(t, 0))
            out.append(str(total + acc))
        else:
            out.extend(str(_UNITS.get(t, _TENS.get(t, 0))) for t in run)
        run.clear()

    for tok in tokens:
        if tok in _NUMWORDS:
            run.append(tok)
        else:
            flush()
            out.append(tok)
    flush()
    return out


def norm(text):
    text = text.lower()
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = re.sub(r"(\d)[,](\d)", r"\1\2", text)    # 4,200 -> 4200
    text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)  # 4th -> 4
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(_digits(text.split()))


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev = cur
    return d[len(h)] / max(len(r), 1)


def tts(text, out):
    subprocess.run([os.path.join(RUNTIME, "llama-tts"), "-m", MODEL, "-mm", MMPROJ,
                    "-ngl", "99", "-t", "6", "-p", text, "-o", out],
                   check=True, capture_output=True)


def asr(path):
    t0 = time.time()
    r = subprocess.run([os.path.join(RUNTIME, "llama-mtmd-cli"), "-m", MODEL,
                        "--mmproj", MMPROJ, "-ngl", "99", "-t", "6", "--audio", path,
                        "-sys", "Perform ASR in English.", "-p", "<__media__>",
                        "--temp", "0"],
                       check=True, capture_output=True, text=True)
    lines = [ln.strip() for ln in r.stdout.splitlines()
             if ln.strip() and not ln.startswith(("llama_", "load_", "print_info", "clip_",
                                                  "mtmd_", "main:", "encode_"))]
    body = "\n".join(lines)
    body = re.sub(r"<think>.*?</think>", " ", body, flags=re.S)
    return body.strip(), time.time() - t0


def main():
    tmpdir = tempfile.mkdtemp(prefix="asr-")
    rows, total = [], []
    print("| phrase | audio | ASR wall | RTF | WER | hypothesis |")
    print("|---|---|---:|---:|---:|---|")
    for name, text in PHRASES:
        wav = os.path.join(tmpdir, f"{name}.wav")
        tts(text, wav)
        hyp, secs = asr(wav)
        e = wer(text, hyp)
        total.append(e)
        with wave.open(wav) as w:
            dur = w.getnframes() / w.getframerate()
        print(f"| {name} | {dur:.1f}s | {secs:.2f}s | {dur / secs:.1f}x | {e:.0%} | "
              f"{hyp.replace('|', '/')} |")
    print()
    print(f"**mean WER: {sum(total) / len(total):.1%}** over {len(total)} phrases "
          f"(synthetic speech, model transcribing its own TTS)")
    print(f"wavs kept in {tmpdir}" if "--keep" in sys.argv else f"wavs in {tmpdir} (rm -r to clean)")


if __name__ == "__main__":
    main()
