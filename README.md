# lfm — LFM3 local inference lab

Fully offline deployment of Liquid AI **LFM3** on this laptop (Intel Core Ultra 7 258V,
Lunar Lake / Arc 130V iGPU / Intel NPU 4, 30 GiB shared LPDDR5X, no dGPU).

Nothing here needs the network at run time. Everything is self-contained under this
directory plus `~/models/lfm3` and `~/Code/lfm/runtimes`.

## Layout

```
~/Code/lfm/
├── runtimes/                       llama.cpp b11985 prebuilt (no system packages touched)
│   ├── ubuntu-vulkan-x64/          Vulkan backend  → Arc 130V iGPU   (default)
│   ├── ubuntu-x64/                 plain CPU
│   └── ubuntu-openvino-2026.4.1-x64/  OpenVINO backend (CPU only for now, see PLAN.md)
├── scripts/                        run / bench entry points
├── testdata/                       test image + generated audio
├── logs/
└── .venv/                          uv venv (huggingface_hub `hf` CLI for downloads)

~/models/lfm3/
├── LFM3-1.7B-GGUF/                 QAD-Q4_0, Q4_K_M, Q8_0
└── LFM3-Omni-2B-GGUF/              Q4_K_M + mmproj-F16
```

Runtimes are official llama.cpp `b11985` release tarballs, unpacked — no apt, no
system libs replaced. Only the preinstalled Vulkan loader + Mesa Intel ICD are used.

## Models

| role | model | file | size |
|---|---|---|---|
| text, default | `LiquidAI/LFM3-1.7B-GGUF` | `LFM3-1.7B-QAD-Q4_0.gguf` | 993 MiB |
| text, safest Q4 | same | `LFM3-1.7B-Q4_K_M.gguf` | 1.02 GiB |
| text, high quality | same | `LFM3-1.7B-Q8_0.gguf` | 1.71 GiB |
| omni (text/image/audio/video → text/speech) | `LiquidAI/LFM3-Omni-2B-GGUF` | `LFM3-Omni-2B-Q4_K_M.gguf` + `mmproj-LFM3-Omni-2B-F16.gguf` | 1.05 GiB + 906 MiB |

`QAD-Q4_0` is Liquid's quantization-aware-distilled 4-bit checkpoint: it is both the
smallest *and* the fastest file here, and it recovers most of Q4_0's accuracy loss.
It is the default for the 1.7B.

License: **LFM Open License v1.0** — free for research / non-commercial, and for
commercial use below USD 10M annual revenue; above that a commercial license is
required. No copyleft.

## Usage

```bash
cd ~/Code/lfm

# text model, OpenAI-compatible API + web UI on :8110
./scripts/serve-text.sh                 # auto: ctx<=4096 -> OpenVINO GPU, else Vulkan
./scripts/serve-text.sh qad vulkan      # force Vulkan (long documents, agent loops)
./scripts/serve-text.sh qad openvino-gpu  # force OpenVINO GPU (2x prefill on short prompts)
./scripts/serve-text.sh qad cpu         # CPU only
FA=on UB=512 ./scripts/serve-text.sh    # override a tuned default, t=6

# omni model (vision + audio in, text + speech out)
./scripts/serve-omni.sh                 # :8111, --reasoning off

# one-shot CLI
./scripts/omni.sh image testdata/cats.png "Describe this image in one sentence."
./scripts/omni.sh audio speech.wav "Perform ASR in English."
./scripts/omni.sh tts "Hello from a fully offline model." testdata/out.wav

# benchmarks
./scripts/bench.sh                      # backend x quant matrix
./scripts/hw-sweep.sh                   # threads / ngl / ub / kv / fa sweep
./scripts/hw-sweep.sh ngl               # one stage only
./scripts/power-bench.sh configs        # tokens per joule (unplug the charger!)
./scripts/power-bench.sh profiles       # power-saver / balanced / performance
bash ./scripts/get-sycl.sh              # fetch the SYCL build for comparison
```

Any OpenAI client works against `http://127.0.0.1:8110/v1`.

Sampling that Liquid recommends:

* text / instruct: `temp 0.1, top_k 50, repetition_penalty 1.1`
* omni text output: greedy (`temp 0`)
* omni speech output: `temp 0.4, audio repetition_penalty 1.2`

Thinking is **on by default** for LFM3-1.7B. Turn it off per request with
`"chat_template_kwargs": {"enable_thinking": false}`.

## Measured on this machine (2026-10-30, llama.cpp b11985, `llama-bench`, `-p 256 -n 64`)

| backend | quant | pp 256 (t/s) | tg 64 (t/s) |
|---|---:|---:|---:|
| CPU, 2 threads (default) | Q4_K_M | 69 | 24.9 |
| CPU, 4 threads | Q4_K_M | 113 | 39.0 |
| **CPU, 6 threads** | Q4_K_M | 139 | **40.7** |
| CPU, 8 threads | Q4_K_M | 156 | 35.3 |
| **Vulkan iGPU** | Q4_K_M | **2142** | 31.8 |
| **Vulkan iGPU** | **QAD-Q4_0** | **2258** | **40.7** |
| Vulkan iGPU | Q8_0 | 2094 | 18.0 |
| CPU 6t | QAD-Q4_0 | 164 | **46.5** |
| CPU 6t | Q8_0 | 87 | 28.1 |
| OpenVINO CPU | Q4_K_M | 168 | 31.4 |

Real server run (Vulkan, QAD-Q4_0, `-fa on`, 8K ctx): pp ~289 t/s, tg ~40 t/s.

### Tuned defaults (after the hardware sweep, 11.7K-token prompt, real server)

| flag | setting | why |
|---|---|---|
| `-fa` | **off** | at a 11.7K-token prompt FA costs 1.8x on prefill (pp 1396 vs 792 t/s; `auto` = 933). Invisible at p=256 — it only shows at depth. |
| `-ub` | **1024** | no measurable difference vs 512/2048 (all inside noise); only `<=128` is clearly worse. Kept at 1024. |
| `-ctk/-ctv` | **f16** (default) | `q8_0` costs 20-40% of tg at depth (32.2 → 19.5 t/s @ d=32768); `q4_0` worse still |
| `-ngl` | **99** | mixed CPU+iGPU splits never win: ngl=24 → tg 38.2/pp 1826, ngl=99 → 37.4/2212 |
| `-t` | **6** | 8 threads regresses (E-core contention + thermals) |
| power profile | **power-saver** (Vulkan) | tg unchanged (39.1 vs 39.1), but 23.8 W vs 32.4 W under `performance`. The box is bandwidth-bound, not power-bound. |

### Multi-turn / prefix caching (the long-context win)

Sending an 11.7K-token document and then asking 4 follow-up questions:

| setup | turn 1 prefill | turns 2-4: tokens re-evaluated | turns 2-4: prefill | 
|---|---|---|---|
| default (`--swa-full` off) | 8.5 s (11731 tok) | 1064-1073 tok | 2.2-2.8 s |
| `--swa-full` | 8.5 s (11731 tok) | **22-49 tok** | **1.4-1.8 s** |
| `--ctx-checkpoints 0` | 7.3 s | **11776-11857 tok (cache disabled)** | 7.2-7.3 s |
| new document (control) | 8.5 s | 11734 tok, no reuse | 8.5 s |

* Prefix caching works, and `--swa-full` is what makes it *cheap*: LFM3's 5 sliding-window
  layers force llama-server to rebuild ~1k tokens of SWA state per turn unless the cache is
  kept at full size. **~2x faster multi-turn, ~0.2% of the document re-processed.**
  Verified up to 128K context; memory cost is device-side and not a problem on 30 GiB shared.
* `--ctx-checkpoints 0` **disables prefix reuse entirely** — never set it.
* `--cache-reuse 0` changes nothing (different mechanism: KV shifting).
* Answers are identical across all variants, so this is a pure speed win.
* `serve-text.sh` / `serve-omni.sh` now pass `--swa-full` by default (`SWA_FULL=0` to disable).

### Thinking on vs off (same 8-task A/B, greedy, `scripts/think_eval.py`)

| case | thinking off | thinking on | tokens off/on | seconds off/on |
|---|---|---|---|---|
| math: change | PASS | PASS | 49/607 | 6.1/26.0 |
| math: arrival time | PASS | PASS | 106/278 | 7.7/29.6 |
| math: ages | PASS | PASS | 84/183 | 6.8/16.0 |
| logic: contrapositive | PASS | PASS | 2/652 | 2.0/32.9 |
| json: invoice | *fail* | PASS | 38/866 | 3.3/106.2 |
| json: order | PASS | PASS | 34/362 | 2.7/12.7 |
| format: exactly 3 words | PASS | PASS | 4/170 | 3.5/24.4 |
| tool call | PASS | PASS | 20/79 | 4.2/11.3 |
| **score** | **7/8** | **8/8** | **337** | **3197 (9.5x)** |

**7.1x wall-clock for +1 task.** The single diff was instruction adherence (thinking mode
returned `"name": "Acme Ltd"`; no-thinking returned `"Invoice from Acme Ltd"`), not a hard
failure. Simple math, logic and tool calling are unaffected.

Keep thinking **off** for agent loops and interactive use; turn it on for one-off messy
extraction. `reasoning_effort` / thinking levels are ignored by llama.cpp — only on/off works.

### Power efficiency (unplugged, median of 0.5 s battery samples, decode 512 tok x2)

| profile | backend | pp (t/s) | tg (t/s) | W | tok/J |
|---|---|---:|---:|---:|---:|
| power-saver | **Vulkan ngl=99** | 2170 | **39.1** | **23.8** | **1.65** |
| balanced | Vulkan ngl=99 | 2188 | 39.1 | 25.0 | 1.57 |
| performance | Vulkan ngl=99 | 2215 | 39.6 | 32.4 | 1.22 |
| power-saver | cpu t=6 | 104 | 34.3 | **16.1** | **2.12** |
| balanced | cpu t=6 | 149 | 39.9 | 25.6 | 1.56 |
| performance | cpu t=6 | 155 | 45.9 | 38.9 | 1.18 |

**`performance` costs ~30% more power for ~1% more tokens** — it is never worth it on this
machine. **`power-saver` + Vulkan keeps full speed** (39.1 t/s, prefill unchanged) at the
lowest whole-machine draw measured for a usable chat setup. `power-saver` + CPU is the
lowest-power option overall (16 W) but prefill collapses to ~104 t/s, so it only suits
decode-heavy / background work.

Screen brightness and background load move these numbers by ±30%, so only compare rows
measured in the same sweep.

Long-context reality check (`-p 2048`, QAD-Q4_0): pp falls from ~2100 t/s at depth 0 to
~760 at d=16384 and ~145 at d=32768. Prefill is the wall for long documents, not tg.

Takeaways:

* **Prompt processing: iGPU wins by ~15x** (2.2k vs 140 t/s). Long prompts, RAG,
  image/video encoding all belong on Vulkan.
* **Token generation is memory-bandwidth bound** and roughly a tie: iGPU ~40 vs
  CPU-6t ~46 on the same QAD-Q4_0 file. Anything that shrinks bytes-read wins tg —
  hence Q4_0 > Q4_K_M > Q8_0.
* **Flash attention and KV quantisation both lose on this iGPU** — dequant/fallback
  overhead outweighs the memory saving. Defaults are set accordingly.
* 6 threads is the CPU sweet spot (8 threads regresses, thermal + E-core contention).
* The machine is on the `balanced` power profile / `powersave` governor; see PLAN.md.
* **Speculative decoding does not work here.** No LFM3 drafter exists (LFM2.5-350M is
  rejected for incompatible vocab), and `--spec-type ngram-*` is ~13% *slower* (code -40%,
  quote -45%). Decode is bandwidth-bound, so extra draft/verify passes cannot pay for
  themselves at 1.7B on a shared-memory iGPU.
* **Concurrency needs flash attention.** 4 slots x 4 workers: 9.3 tok/s aggregate with
  `-fa off` vs **43.2 tok/s with `-fa on`** (no crashes either way). FA is only bad for the
  single-stream long-prompt case, which is the intended laptop use — hence the `-fa off` default.
* Knob-level differences below ~10% are noise on this box (shared-memory LPDDR5X, thermal
  drift); `hw-sweep.sh` uses `-r 2`, so only treat double-digit deltas as real.

Omni-2B (Q4_K_M, Vulkan): image encode ~1.0 s for a 662 KB PNG; TTS runs at
0.83x real time (5.6 s of 24 kHz audio in 6.7 s wall).

## Known limitations

* **Traditional Chinese is weak — measured, not guessed.** On 400 TMMLU+ items LFM3-1.7B
  scores **32.0%** (chance 25%), behind Qwen3.5-0.8B (**37.8%**) at half the size and
  Qwen3.5-2B (**45.2%**). Q8_0 scored 27.8%, so this is missing knowledge, not
  quantisation loss. Glyphs are correct 繁體 but vocabulary drifts mainland (光刻 vs 微影).
  Treat it as an English-first tool; do not use it as a zh-TW assistant. See PLAN.md 6a.
* LFM3-1.7B is **not** for agentic coding or long multi-step reasoning (vendor's own
  statement). Strong at instruction following, JSON/structured output, tool calling.
* Omni audio input is marked **experimental** by llama.cpp ("may have reduced quality").
* **OpenVINO is device-dependent.** With `intel-opencl-icd` + `libze1` installed, the GPU
  device gives **4190-4296 t/s prefill at 512 tokens and 48 t/s decode** (Vulkan: 2087 / 37) at
  the same tok/J (2.86). It loses at long context (11.7K: pp 993 vs 1156, tg 23 vs 31) and its
  prefix cache re-evaluates 516 tokens per turn where `--swa-full` on Vulkan needs 4. The
  OpenVINO **CPU** device is the broken one (>512 tokens) — that was the earlier "dead end".
  Try it with `GGML_OPENVINO_DEVICE=GPU ./scripts/serve-text.sh qad openvino 8110`.
* **The NPU is not worth using.** `Intel(R) AI Boost` works after installing Intel driver 1.38,
  Level Zero 1.32 and a udev trigger, but measures **26 t/s / 2.41 tok/J** — slower *and* less
  efficient than the iGPU — with a hard **~768 token context ceiling**.
* Wiki: LFM3's tool-call format is Qwen3-Coder-style XML, needs `--jinja`
  (already passed by the scripts) and a `qwen3_coder` parser downstream.
