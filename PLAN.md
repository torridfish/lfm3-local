# PLAN — LFM3 local deployment: experiment log and remaining work

> Documentation convention for this repo: English. Conversation can be in Chinese.

Machine: Intel Core Ultra 7 258V (Lunar Lake), Arc 130V iGPU, Intel NPU 4, 30 GiB shared
LPDDR5X, 60.35 Wh battery, Ubuntu 26.04, no dGPU.
Model under test unless stated: **LFM3-1.7B, QAD-Q4_0**, llama.cpp **b11985**, Vulkan backend.

## Goal

Not maximum speed. This is a laptop: find the point where **power draw is low and speed is
still good**, and make long-context / agent use actually usable offline.

Every experiment records: **pp t/s, tg t/s, correctness** (fixed battery of zh-TW questions,
JSON extraction, one tool-call task), and for power work **W / tok/J**.

Tools: `./scripts/bench.sh`, `./scripts/hw-sweep.sh`, `./scripts/power-bench.sh`,
`./scripts/prefix-test.sh`, `./scripts/think_eval.py`. Results land in `logs/*.md`.

---

## Progress dashboard

| # | Experiment | Status | Outcome |
|---|---|---|---|
| 0 | Power profile (power-saver / balanced / performance) | **done** | `performance` = +30% W for +1% t/s. **Use `power-saver` with Vulkan.** |
| 2 | Quantisation matrix (Q4_0 / Q4_K_M / Q8_0) | **done** | **QAD-Q4_0 wins on both size and tg** (40.7 vs 31.8 vs 18.0 t/s). |
| 4 | Flash attention on/off | **done** | **`-fa off` is 2.8x faster on prefill at depth.** FA costs 1.8x at 11.7K tokens. |
| 4 | KV cache quantisation (q8_0 / q4_0) | **done — rejected** | tg at d=32768: 32.2 (f16) → 19.5 (q8_0). Dequant costs more than the bandwidth saved. |
| 5 | `-ngl` mixed CPU+iGPU split | **done — rejected** | ngl=24 → tg 38.2 / pp 1826 vs ngl=99 → 37.4 / 2212. Splitting never wins. |
| 5 | `-ub` / `-b` sweep | **done** | 512-2048 all within noise; `<=256` clearly worse. Default `-ub 1024`. |
| 5 | `--parallel` soak (crash + scaling) | **done** | No crash. But 4 slots with `-fa off` = 9.3 tok/s vs **43.2 tok/s with `-fa on`**. |
| 5 | Prefix caching / `--swa-full` | **done — big win** | Per-turn re-eval **1070 → 22 tokens**; multi-turn prefill **2.2-2.8 s → 1.4-1.8 s**. Now default. |
| 8 | Power efficiency (tok/J) | **done** | 23.8 W / 39.1 t/s / **1.65 tok/J** at power-saver+Vulkan. ~2.5 h on battery. |
| 6 | Thinking on vs off | **done** | Score 7/8 → 8/8, but **9.5x tokens, 7.1x wall**. Off for agents. |
| 1 | Intel OpenVINO | **done — works, device-dependent** | GPU: 2x prefill / 1.3x decode at short prompts, same tok/J; loses at long context. CPU device useless. |
| — | pi coding-agent integration | **done** | `lfm3-local` provider on :8110, tool calling verified end-to-end. |
| 7 | Omni: vision / TTS / audio | **partly done** | vision OK, TTS 0.83x realtime, audio marked experimental by llama.cpp. |
| 6 | Traditional Chinese evaluation (TMMLU+) | **done** | **32.0% vs Qwen3.5-0.8B 37.8% / 2B 45.2%. Not a zh-TW model.** |
| 1 | OpenVINO GPU (`intel-opencl-icd` + `libze1`) | **done** | 4190-4296 t/s prefill @512; 516 tokens re-evaluated per cached turn vs Vulkan's 4. |
| 1 | OpenVINO NPU (driver not in Ubuntu repos) | **not started, low priority** | Expected to save power, not to be faster. |
| 3 | Speculative decoding (ngram / drafter) | **done — dead end** | ngram variants are 13% slower; no LFM3 drafter exists and LFM2.5-350M has an incompatible vocab. |
| 7 | Omni mmproj quantisation, `--image-max-tokens` | **done** | mmproj Q4 = same speed, -627 MB. Image-token flag has no measurable effect (and CLI only). |
| — | SYCL backend comparison | **blocked: SYCL 9 vs 8 ABI clash** | Needs Intel oneAPI repo; Ubuntu's oneDNN-SYCL targets SYCL 8. Details in the SYCL section. |
---

## 0. Power profile — done, do not use `performance`

| item | balanced | performance |
|---|---|---|
| Vulkan QAD-Q4_0 pp | 2258 | 2274 |
| Vulkan QAD-Q4_0 tg | 40.7 | 40.5 |
| CPU t=6 tg | 46.5 | 45.7 |

`performance` buys nothing measurable and costs ~30% more watts. The machine is
**bandwidth-bound, not power-bound** — good news for a laptop.

Which profile to use depends on the backend: **Vulkan + `power-saver`** (same speed, lowest
draw); **CPU-only needs `balanced`** (power-saver halves prefill). Full data in section 8.

## 1. Intel OpenVINO

### 1a. State (measured)

`runtimes/ubuntu-openvino-2026.4.1-x64/` (official prebuilt) only exposes CPU:

```
OPENVINO0: GGML_OPENVINO_DEVICE=CPU (selected) - Intel(R) Core(TM) Ultra 7 258V
GGML_OPENVINO_DEVICE=GPU is not available. Set it to one of the available OpenVINO devices:
  CPU: Intel(R) Core(TM) Ultra 7 258V
```

Cause pinned down: the tarball ships `libopenvino_intel_gpu_plugin.so` and
`libopenvino_intel_npu_plugin.so`, but the system has no matching device driver —
`/etc/OpenCL/vendors/` contains only `nvidia.icd` (no `intel.icd`), and there is no
level-zero NPU driver (`libze_intel_npu.so`). No driver → plugin load fails → CPU only.

OpenVINO CPU measured: pp 232 / tg 32.8 (QAD-Q4_0) — **slower than plain CPU t=6**.

### 1b. To enable GPU / NPU (not done — needs apt approval)

| target | package | note |
|---|---|---|
| OpenVINO GPU (Arc 130V) | `intel-opencl-icd` | provides `/etc/OpenCL/vendors/intel.icd`; present in Ubuntu 26.04 (26.05.37020.3-1) |
| OpenVINO NPU (NPU 4) | `intel-level-zero-npu` | **not in Ubuntu repos** — needs Intel's own apt repo |
| diagnostics | `clinfo`, `vulkan-tools` | confirm the device appears |

```bash
sudo apt install -y intel-opencl-icd clinfo
clinfo -l                                  # expect Intel(R) Arc(TM) Graphics
GGML_OPENVINO_DEVICE=GPU ./runtimes/ubuntu-openvino-*/llama-cli --list-devices
```

Expectation management: **the NPU will not make decode faster.** It shares the same
LPDDR5X, and decode is bandwidth-bound. NPU value is power and always-on availability.

### 1b-bis. OpenVINO after `intel-opencl-icd` + `libze1` were installed — **it works, and it is fast**

With the Intel OpenCL ICD in place the bundled runtime enumerates two devices:

```
OPENVINO0: GGML_OPENVINO_DEVICE=CPU - Intel(R) Core(TM) Ultra 7 258V
OPENVINO1: GGML_OPENVINO_DEVICE=GPU (selected) - Intel(R) Arc(TM) Graphics (iGPU)
```

`GGML_OPENVINO_DEVICE=GPU` gives a usable, fast backend. `NPU` is still absent (no driver
in Ubuntu's repos).

| workload | Vulkan | **OpenVINO GPU** | OpenVINO CPU |
|---|---:|---:|---:|
| pp 512 (short prompt) | 2087 | **4190-4296** | fails |
| tg, short prompt | 37 | **48** | — |
| pp 11.7K (cold) | **1156** | 993 | — |
| tg @ 11.7K | **31-32** | 23-24 | — |
| cached turn over a 11.7K doc | **4 tokens re-evaluated** (`--swa-full`) | 516 tokens re-evaluated | — |
| decode power (512 tok) | 12.6 W | 13.3 W | 11.9 W |
| tok/J (decode) | **2.86** | **2.86** | 0.92 |

The earlier "dead end" verdict was **CPU-device specific**. Corrected picture:

* **Short/medium prompts: OpenVINO GPU wins clearly** — 2x prefill, ~1.3x decode, same tok/J.
* **Long context: Vulkan still wins** — better cold prefill, 1.3x decode, and `--swa-full`
  makes cached turns nearly free (4 vs 516 tokens re-evaluated per turn).
* The OpenVINO CPU device is the useless one (dies above ~512 tokens, worst tok/J).
* Default stays Vulkan because the intended workload is long documents + agent loops.
  OpenVINO GPU is a genuine option for short-interaction use:
  `GGML_OPENVINO_DEVICE=GPU ./scripts/serve-text.sh qad openvino 8110`

### 1c. NPU (`Intel(R) AI Boost`) — enabled, measured, **not worth using**

Getting it running took four separate fixes; each one masked the next, so the notes below
are the useful part.

| step | symptom | fix |
|---|---|---|
| 1 | no userspace driver, no package in Ubuntu repos | installed Intel driver **v1.38.0** from the GitHub release tarball (`ubuntu2604`), no apt repo needed: `intel-level-zero-npu`, `intel-driver-compiler-npu`, `intel-fw-npu` + `libtbb12` |
| 2 | `ZE_RESULT_ERROR_UNINITIALIZED`, loader reported `ze_loader.dll version: 1.28.2` and `zeInitDrivers not supported` | upgraded `libze1` to **1.32.0** (the pairing Intel documents for driver 1.38) |
| 3 | module unloaded itself, `/dev/accel/` disappeared | `sudo modprobe intel_vpu` |
| 4 | device came back as `crw------- root root`, no ACL → still uninitialized | the uaccess rule in `/usr/lib/udev/rules.d/70-uaccess.rules` had not fired: `sudo udevadm trigger --subsystem-match=accel --action=add` |

After that OpenVINO reports three devices and the NPU is selectable:

```
OPENVINO0: CPU - Intel(R) Core(TM) Ultra 7 258V
OPENVINO1: GPU - Intel(R) Arc(TM) Graphics (iGPU)
OPENVINO2: NPU (selected) - Intel(R) AI Boost (NPU 4000)
```

Measured (`llama-bench`, QAD-Q4_0):

| backend | pp 512 | tg | power (400 tok decode) | tok/J |
|---|---:|---:|---:|---:|
| Vulkan | 2087 | 36 | 12.6 W | 2.86 |
| **OpenVINO GPU** | **4190** | **41** | **11.4 W** | **3.60** |
| OpenVINO NPU | 1135 | 26 | 10.8 W | 2.41 |

**The NPU loses on every axis.** It is 2.4x slower than the iGPU at prefill, 1.6x slower at
decode, and *less* efficient per joule — so the energy-efficiency case for an NPU does not
survive contact with a shared-memory LLM workload. On top of that:

* **Context ceiling ~768 tokens.** `-c 512` works (26 t/s), `-c 768` works but collapses to
  14 t/s, `-c 1024` fails with `Compute error` at `ub = 64` as well, so it is a graph-size
  limit, not a batching one. Half a page of context is not a usable assistant.
* `llama-server` only works with `-b 512 -ub 512 -c 512`; defaults (2048) die immediately.
* `llama-cli` on NPU printed nothing and exited; only the server path is usable.

Also note the `-b/-ub/-c 512` constraint silently caps every backend in the table above, which
is why these tg numbers are lower than the long-context runs elsewhere in this file.

**Verdict: keep the driver installed (it costs nothing and `intel_vpu` is now healthy), but do
not select the NPU.** Revisit only if OpenVINO ships a static-shape LLM path for it (compiled
blob), which llama.cpp's per-op backend cannot provide.

## 2. Quantisation — QAD-Q4_0 is the one to use

| quant | size | Vulkan pp | Vulkan tg | CPU t=6 tg |
|---|---:|---:|---:|---:|
| **QAD-Q4_0** | 993 MiB | 2258 | **40.7** | **46.5** |
| Q4_K_M | 1.02 GiB | 2142 | 31.8 | 40.7 |
| Q8_0 | 1.71 GiB | 2094 | 18.0 | 28.1 |

Q4_0 kernels are the most mature in ggml, and Liquid's QAD (quantisation-aware distillation)
checkpoint recovers most of Q4_0's accuracy loss — so the smallest file is also the fastest.
Q8_0 only if accuracy is demonstrably insufficient.

Not yet done: compare QAD-Q4_0 vs Q8_0 output quality on the fixed task battery.

## 3. Speculative decoding — **measured, dead end for this model**

`scripts/spec_eval.py`, QAD-Q4_0 on Vulkan, 4 workloads, temperature 0, 160 max tokens:

| setup | chat | json | code | quote | mean |
|---|---:|---:|---:|---:|---:|
| **baseline** | 33.2 | 34.9 | **35.1** | **35.6** | **34.7** |
| `--spec-type ngram-mod` | 34.6 | 32.9 | 21.0 | 32.4 | 30.2 |
| `--spec-type ngram-simple` | 34.2 | 35.1 | 35.5 | 19.7 | 31.1 |
| LFM2.5-350M as drafter | (vocab incompatible — refused, see below) |  |  |  |  |

Findings:

1. **No compatible drafter exists.** Hugging Face has no `LFM3-1.7B-drafter` (only FP8 / MLX /
   ONNX / NVFP4 variants of the same weights). Using `LiquidAI/LFM2.5-350M` instead is
   rejected outright:
   ```
   W spec: draft model bos tokens must match target model to use speculation
   E spec: the target and draft vocabs are not compatible
   E srv load_model: failed to initialize speculative decoding context
   ```
   The server still starts, which silently gives you baseline speed — check the log before
   believing any drafter numbers.
2. **ngram / lookup speculation makes things worse** (-13% mean), and the damage is
   workload-dependent: `code` drops 40% with `ngram-mod`, `quote` drops 45% with
   `ngram-simple`. Verification overhead is not repaid because acceptance on this model is
   low.
3. Root cause is the same one that explains everything else here: decode is
   **bandwidth-bound at ~35-40 t/s**, so every speculative token still costs a drafter pass
   plus a verification pass over the 1 GB of weights. Speculation pays when the target model
   is compute-bound and big; at 1.7B on a shared-memory iGPU it does not.
4. The `drafter/` download (LFM2.5-350M QAD-Q4_0, 210 MB) was deleted after this test.

**Conclusion: leave speculation off. This is the one optimisation the vendor's own
benchmark table cannot deliver on this hardware.**

## 4. Flash attention and KV cache — both rejected

| setting | @ d=16384 | @ d=32768 |
|---|---|---|
| `-fa on` | pp 266 / tg 36.0 | pp 145 / tg 32.2 |
| `-fa off` | **pp 756** / tg 32.8 | — |
| KV `q8_0` | pp 358 / **tg 28.0** | pp 183 / **tg 19.5** |

Real server, 11.7K-token prompt: `-fa off` pp 1396 t/s, `-fa on` 792, `-fa auto` 933.
So flash attention is **1.8x slower** here, and KV quantisation costs 20-40% of tg at depth.

The real long-context wall is **prefill**: 2100 t/s at depth 0, 760 at d=16384, 145 at d=32768.

## 5. Batching, slots, prefix caching

* `-ub`: 512/1024/2048 inside noise, `<=128` clearly worse. Default 1024.
* `-ngl 99` always: mixed splits never win.
* **Prefix caching — the one big win.** 11.7K-token document, 4 follow-up questions:

| setup | turn 1 prefill | turns 2-4 tokens re-evaluated | turns 2-4 prefill |
|---|---|---|---|
| default | 8.5 s (11731) | 1064-1073 | 2.2-2.8 s |
| **`--swa-full`** | 8.5 s | **22-49** | **1.4-1.8 s** |
| `--ctx-checkpoints 0` | 7.3 s | 11776-11857 (cache fully disabled) | 7.2-7.3 s |
| control (new document) | 8.5 s | 11734 (no reuse) | 8.5 s |

  LFM3's 5 sliding-window layers force llama-server to rebuild ~1024 tokens of SWA state per
  turn unless the cache is kept at full size. `--swa-full` removes that; answers are identical
  across all variants (pure speed win). Verified up to 128K context; memory cost is
  device-side. Now the default in `serve-text.sh` / `serve-omni.sh` (`SWA_FULL=0` to disable).
  **Never set `--ctx-checkpoints 0`** — it disables prefix reuse entirely.

### 5a. Concurrency soak — no crash, but a surprise: FA flips the picture

`scripts/soak.sh` / `soak.py`, 22-token answers, QAD-Q4_0, ctx 8192:

| setup | aggregate | per-request latency p50 | failures |
|---|---:|---:|---:|
| 1 slot, fa off | 10.8 tok/s | 1.95 s | 0/8 |
| 4 slots, 4 workers, **fa off** | **9.3 tok/s** | 9.33 s | 0/12 |
| 4 slots, 4 workers, fa off, no `--swa-full` | 9.3 tok/s | 8.21 s | 0/12 |
| 4 slots, 4 workers, **fa on** | **43.2 tok/s** | **1.80 s** | 0/12 |

* **No crash, no `GGML_ASSERT`, no abort** in any run — the LFM2 short-conv concurrency bug
  (#14670) did not reproduce for LFM3. Server healthy after every soak.
* Concurrency is what exposes FA: 4 slots without flash attention collapse to single-stream
  throughput (9.3 tok/s) because batched attention gets padded per sequence. With `-fa on`
  the same load runs **4.6x faster in aggregate**.
* This does **not** contradict section 4: FA is slower for *single-stream, long-prompt*
  prefill (11.7K tokens: 1396 vs 792 t/s). The two regimes differ:
  **single user + long documents → `-fa off`; multiple concurrent clients → `-fa on`.**
  Our default stays `-fa off` because the intended use is one person on a laptop.

## 6. Traditional Chinese and thinking mode

### 6a. zh-TW on TMMLU+ — **done. LFM3 loses to a 0.8B model.**

Harness: `scripts/zh_eval.py` on `ikala/tmmluplus` (67 subjects available on the hub).
10 subjects x 40 questions = **400 items**, seed 1234, scored by the probability the model
assigns to the answer letter in one forward pass (`max_tokens=1, logprobs, top_logprobs`).
Chance = 25%.

| model | file size | TMMLU+ (400 q) | vs chance |
|---|---:|---:|---:|
| Qwen3.5-2B Q4_K_M | 1.2 GB | **45.2%** | +20.2 |
| Qwen3.5-0.8B Q4_K_M | 508 MB | **37.8%** | +12.8 |
| **LFM3-1.7B QAD-Q4_0** | 993 MB | **32.0%** | +7.0 |
| LFM3-1.7B Q8_0 (quant control) | 1.71 GB | **27.8%** | +2.8 |

Per subject, LFM3 (Q4_0) vs Qwen3.5-2B:

| subject | LFM3 | Qwen 2B |
|---|---:|---:|
| computer_science | 40.0% | 52.5% |
| physics | 20.0% | 42.5% |
| junior_chemistry | 35.0% | 37.5% |
| junior_math_exam | 20.0% | 32.5% |
| junior_science_exam | 27.5% | 50.0% |
| junior_social_studies | 42.5% | 62.5% |
| introduction_to_law | 30.0% | 40.0% |
| economics | 32.5% | 42.5% |
| geography_of_taiwan | 37.5% | 52.5% |
| traditional_chinese_medicine | 35.0% | 40.0% |

Findings:

1. **LFM3-1.7B scores 32.0% on a Traditional Chinese knowledge benchmark — 7 pp above
   guessing — and is beaten by Qwen3.5-0.8B (37.8%) at half the size.** On `physics` it is
   below chance (20%).
2. **Quantisation is not the cause.** Q8_0 scored 27.8%, *lower* than Q4_0 (SE ≈ 2.2 pp at
   n=400, so this is a wash, not a regression). The model simply lacks zh-TW knowledge.
3. **Format compliance is also weaker.** Re-scoring a 2-subject subset (80 items) by
   generation instead of letter-probability: LFM3 30.0% → **18.8%**, Qwen3.5-2B 42.5% → 40.0%.
   LFM3 frequently fails to answer with a bare letter, which matters for any structured use.
4. This confirms the model card's own signal (`zh` with no S/T split, no Chinese
   specialised variant, C-Eval estimate ~7%).

Caveats: letter-probability scoring is not the official TMMLU+ generation protocol, so the
**absolute** numbers are not comparable to published TMMLU+ leaderboards — the relative
ranking is. n=400 gives SE ≈ 2.2 pp, so treat sub-4 pp gaps as noise. Raw outputs in
`logs/eval/`.

**Decision: LFM3 is an English-first tool here. Do not use it as a zh-TW assistant.**
Acceptable zh-TW uses: formatting/translating short strings, summarising to a schema,
where a wrong word is visible and cheap to fix.

### 6b. Thinking on vs off — `scripts/think_eval.py`, 8 deterministic tasks

| case | off | on | tokens off/on | seconds off/on |
|---|---|---|---|---|
| math: change | PASS | PASS | 49/607 | 6.1/26.0 |
| math: arrival time | PASS | PASS | 106/278 | 7.7/29.6 |
| math: ages | PASS | PASS | 84/183 | 6.8/16.0 |
| logic: contrapositive | PASS | PASS | 2/652 | 2.0/32.9 |
| json: invoice | *fail* | PASS | 38/866 | 3.3/**106.2** |
| json: order | PASS | PASS | 34/362 | 2.7/12.7 |
| format: exactly 3 words | PASS | PASS | 4/170 | 3.5/24.4 |
| tool call | PASS | PASS | 20/79 | 4.2/11.3 |
| **score** | **7/8** | **8/8** | **337** | **3197 (9.5x)** |

The single diff was instruction adherence (`"name": "Acme Ltd"` vs `"Invoice from Acme Ltd"`),
not a reasoning failure. 7.1x wall clock for one borderline task.
**Thinking off for agents and interactive use; on only for one-off messy extraction.**
`reasoning_effort` and pi's thinking levels are ignored by llama.cpp — only on/off exists.

## 7. Omni-2B — works, per-feature status

* Weights: `LFM3-Omni-2B-Q4_K_M.gguf` (1.05 GiB) + projector.
* **Vision: working** — correctly described a test image.
* **TTS: working** — `llama-tts`, 24 kHz mono, 5.6 s of audio in 6.7 s (0.83x realtime).
* **Audio input: experimental** (llama.cpp's own warning); ASR needs real speech samples,
  and there is no TTS/`espeak` installed to generate them — batch-testing blocked.
* `llama.cpp` also exposes `/v1/realtime` (WebSocket voice conversation with tool calls) —
  feature verification, not an optimisation.

### 7a. Projector quantisation — memory win, not a speed win

`llama-mtmd-cli`, one 662 KB PNG, `Describe this image in one sentence.`, temp 0:

| mmproj | size | encode | answer |
|---|---:|---:|---|
| F16 | 906 MB | 222 ms | correct (two cats on fishing net) |
| Q8_0 | 484 MB | 445 ms | correct, identical wording |
| Q4_K_M | 279 MB | 330 ms | correct, slightly different wording |

Encoding is 200-450 ms in every case — inside noise — and only a few percent of a full
vision request. So **quantising the projector buys 627 MB of RAM/disk, not latency**.
On a 30 GiB shared-memory laptop that is still worth it after the swap incident:
`env.sh` now prefers `mmproj-...-Q4_K_M.gguf` when present, falling back to F16.

Caveat: one image, three variants. Quality equivalence is plausible (all descriptions are
correct) but not established across a test set.

### 7a-bis. ASR round-trip — **works well; only alphanumeric names fail**

No human speech samples exist here (no `espeak`/`piper`/`sox`, and installing them needed apt),
so `scripts/asr_eval.py` synthesises each phrase with the model's own TTS head and transcribes
it back. Synthetic speech is cleaner than a room, so read this as an **upper bound**.

| phrase | audio | ASR wall | RTF | WER |
|---|---:|---:|---:|---:|
| plain sentence | 5.3 s | 3.42 s | 1.5x | **0%** |
| numbers / dates / room | 4.3 s | 3.23 s | 1.3x | 8% |
| tech identifiers | 8.3 s | 3.51 s | 2.4x | 33% |
| homophones (read/red, write/review) | 3.2 s | 3.09 s | 1.0x | **0%** |
| 30-word technical sentence | 12.5 s | 3.82 s | 3.3x | 3% |
| place names | 4.2 s | 3.12 s | 1.3x | **0%** |

* Accuracy on ordinary English is effectively perfect, including homophones and punctuation.
* **Alphanumeric identifiers break it**: "Arc 130V" came back as "Arcs Luntrain V" and
  "LPDDR5X" as "LPDR5X" — expected for a 2B model, but it rules out ASR of model numbers,
  part codes, or error IDs without a post-pass.
* Throughput is ~3-4x realtime on long clips, but there is a **~3 s fixed cost per CLI call**
  (model + projector load, audio encode). Short utterances are dominated by it; a server
  would be the right shape for an assistant.
* Scorer notes: `norm()` maps spoken numbers to digits ("March fourth" -> "march 4",
  "eighty five hundred" -> "8500") and strips ordinal suffixes, otherwise correct
  transcriptions score as errors. First run reported 14.6% mean WER purely from formatting.

### 7b. Image token limits — CLI only, no measurable effect

`--image-max-tokens` exists in `llama-mtmd-cli` but **not** in `llama-server` (so not
settable for the API path). Sweeping 256 / 512 / 1024 changed nothing measurable
(encoding 296-451 ms, all inside noise), which suggests the model's fixed visual grid
dominates. Not worth pursuing.

## 8. Power efficiency (tok/J) — the answer for a laptop

Unplugged, median of 0.5 s battery samples, decode 512 tokens x2:

| profile | backend | pp (t/s) | tg (t/s) | W | tok/J |
|---|---|---:|---:|---:|---:|
| power-saver | **Vulkan ngl=99** | 2170 | **39.1** | **23.8** | **1.65** |
| balanced | Vulkan ngl=99 | 2188 | 39.1 | 25.0 | 1.57 |
| performance | Vulkan ngl=99 | 2215 | 39.6 | 32.4 | 1.22 |
| power-saver | cpu t=6 | 104 | 34.3 | **16.1** | **2.12** |
| balanced | cpu t=6 | 149 | 39.9 | 25.6 | 1.56 |
| performance | cpu t=6 | 155 | 45.9 | 38.9 | 1.18 |

1. `performance` is never worth it.
2. **`power-saver` + Vulkan is the sweet spot**: speed identical to balanced (39.1), prefill
   identical, lowest whole-machine draw. The iGPU is not throttled much at low power because
   the CPU only orchestrates.
3. `power-saver` + CPU has the best tok/J (2.12) but prefill collapses to 104 t/s — only for
   pure generation, not chat over documents.
4. Battery: 60.35 Wh → ~2.5 h of continuous inference at 23.8 W.

Measurement caveat: battery readings move ±30% with screen brightness and background load;
**only compare rows from the same sweep.** The script now records the profile in every row,
uses the median, and decodes 512 tokens per measurement.

### 8a. Interleaved A/B: Vulkan vs OpenVINO GPU (short context)

The whole-machine numbers above were the first sweep out of date the moment the GPU device
became usable. Re-measured interleaved (vulkan, ov-gpu, vulkan, ov-gpu, ... x3) at
`-c 512 -b 512 -ub 512`, 400-token decode, power-saver:

| backend | tg per rep (t/s) | W per rep | median tg | median W | tok/J |
|---|---|---|---:|---:|---:|
| Vulkan (`-fa off --swa-full`) | 37, 35, 36 | 9.5, 12.6, 14.8 | 36 | 12.6 | 2.86 |
| **OpenVINO GPU** | 43, 40, 41 | 9.2, 11.4, 11.5 | **41** | **11.4** | **3.60** |

Interleaving matters: single-shot rows differ by up to 55% (9.5 vs 14.8 W on the *same*
backend), so the earlier "GPU and Vulkan are equal on tok/J" reading was noise. Paired
comparison says **OpenVINO GPU is +14% decode and +26% tok/J at short context.**

### 8b. Where exactly the crossover is — `serve-text.sh` now picks automatically

Same doc sliced to 1K/2K/4K/8K tokens, one question each, run twice (cold, then cached).
The first row of each backend is contaminated by Vulkan shader compilation / graph build, so
read from ~2K down:

| prompt | Vulkan cold pp | Vulkan tg | OV GPU cold pp | OV GPU tg | cached re-eval (V / OV) |
|---:|---:|---:|---:|---:|---|
| ~1024 | (418, warm-up) | 36 | (836, warm-up) | 30 | 4 / 516 tok |
| ~2048 | 861 | 30 | **1829** | 27 | 4 / 516 tok |
| ~4096 | 1194 | 31 | 1583 | 22 | 4 / 516 tok |
| ~8192 | **1176** | **32** | 390 | 20 | 4 / 516 tok |

Time to a 48-token answer (prefill + decode):

| prompt | Vulkan | OpenVINO GPU |
|---:|---:|---:|
| 2K | 4.0 s | **2.9 s** |
| 4K | 5.0 s | 4.8 s (tie) |
| 8K | **8.5 s** | 23.4 s |

So the split is **~4K**: GPU wins first-token latency below it because prefill is 2x faster,
and loses above it because its prefill collapses (390 t/s at 8K) and its prefix cache always
re-evaluates 516 tokens. `scripts/serve-text.sh` defaults to `auto` and applies exactly this:

```
AUTO_SPLIT=4096   ctx*slots <= 4096 -> openvino-gpu
                  ctx*slots >  4096 -> vulkan
```

Override with `./scripts/serve-text.sh qad vulkan|openvino-gpu|openvino-npu|cpu` or
`AUTO_SPLIT=2048`. `openvino-npu` is accepted but forces `ctx 512` and prints why.

## 9. pi coding-agent integration

* `~/.pi/agent/models.json` (symlink into the dotfiles repo) has provider **`lfm3-local`**,
  model id `lfm3`, `openai-completions`, baseUrl `http://127.0.0.1:8110/v1`,
  context 65536, max-out 4096, `reasoning: false`.
* Verified: `/v1/models`, tool calling (`finish_reason: tool_calls`, correct args), and a real
  agent loop — `pi -p --provider lfm3-local --model lfm3 "Read note.txt …"` used the read tool
  and answered correctly.
* Measured through pi: ~31 t/s decode, ~1 s prompt eval on cache hits.
* Vendor position, quoted from the card: *"not recommended for agentic coding or complex
  multi-step reasoning due to its small size"* — but the same card says LFM3 was post-trained
  for agentic harnesses *including Pi*. Reading/grep/shell/single-step tool calls work;
  multi-file edits and long self-correcting loops do not.
* Expected pain: per-turn latency 5-8x with thinking on; pi's system prompt + tools cost a
  full prefill on the first request.

## Remaining work, in priority order

| # | Task | Cost | Why |
|---|---|---|---|
| 4 | Speculative decoding (ngram first, then drafter) | 0.5-1 day | biggest remaining tg opportunity (deferred by user) |
| 7 | NPU driver from Intel's repo | 1 day | lowest priority; power/always-on, not throughput |
| 8 | ASR batch test for Omni audio | blocked | needs real speech samples |

### SYCL: what actually blocks it (recorded so nobody repeats the search)

The prebuilt `llama-b11985-bin-ubuntu-sycl-fp16-x64` links against these, and each was
resolved without touching the system where possible:

| library | source that worked |
|---|---|
| `libiomp5.so` | pip `intel-openmp` |
| `libsvml.so`, `libimf.so`, `libirng.so`, `libintlc.so.5` | pip `intel-cmplr-lib-rt` |
| `libsycl.so.9` | pip `intel-sycl-rt` |
| `libmkl_sycl_blas.so.6` + MKL core | pip `onemkl-sycl-blas` |
| `libze_loader.so.1` | apt `libze1` |
| `libdnnl.so.3` (SYCL variant) | apt `libdnnl-sycl3`, extracted rootless via `apt-get download` + `dpkg-deb -x` |

With all of that in place the Arc device is detected, but the link still fails:
`libggml-sycl.so: undefined symbol: dnnl_sycl_interop_engine_create` — llama.cpp needs
oneDNN built for **SYCL 9**, Ubuntu's `libdnnl-sycl3` targets **SYCL 8** (`libsycl.so.8`).
That is an ABI clash, not a missing file. The clean fix is Intel's oneAPI apt repo (matched
SYCL 9 + oneDNN + MKL) — a bigger system change than the OpenVINO GPU route, which already
covers the same question.

## Out of scope / will not do

* No apt changes without explicit approval; nothing that overrides system Mesa/OpenCL.
* No shell rc edits, no touching other `~/Code` projects.
* No deleting models or runtimes without asking.

## Measurement discipline

* Noise floor on this box is roughly ±10% (shared bandwidth, thermal drift, background load).
  `hw-sweep.sh` uses `-r 2` — treat only double-digit deltas as real, or re-run interleaved
  with `-r 5`.
* Power numbers: median of 0.5 s samples, unplugged, same brightness, same sweep.
* Always check correctness alongside speed — several "optimisations" here changed nothing
  except numbers (`-ub`), and `--swa-full` had to be verified to produce identical answers.
