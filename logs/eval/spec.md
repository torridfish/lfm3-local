### baseline 

| workload | tg (t/s) | tokens | wall (s) |
|---|---:|---:|---:|
| chat (free-form) | 33.2 | 160 | 7.04 |
| json (structured) | 34.9 | 57 | 3.53 |
| code (repetitive) | 35.1 | 22 | 2.13 |
| quote (verbatim) | 35.6 | 54 | 3.03 |
| **mean** | **34.7** | | |

### ngram-mod --spec-type ngram-mod

| workload | tg (t/s) | tokens | wall (s) |
|---|---:|---:|---:|
| chat (free-form) | 34.6 | 160 | 6.09 |
| json (structured) | 32.9 | 57 | 4.03 |
| code (repetitive) | 21.0 | 22 | 3.64 |
| quote (verbatim) | 32.4 | 54 | 4.78 |
| **mean** | **30.2** | | |

### ngram-simple --spec-type ngram-simple

| workload | tg (t/s) | tokens | wall (s) |
|---|---:|---:|---:|
| chat (free-form) | 34.2 | 160 | 6.22 |
| json (structured) | 35.1 | 57 | 4.62 |
| code (repetitive) | 35.5 | 22 | 2.52 |
| quote (verbatim) | 19.7 | 54 | 4.88 |
| **mean** | **31.1** | | |

### draft-lfm25-350m --spec-type draft-simple -md ~/models/lfm3/drafter/LFM2.5-350M-QAD-Q4_0.gguf --spec-draft-n-max 6

| workload | tg (t/s) | tokens | wall (s) |
|---|---:|---:|---:|
| chat (free-form) | 37.6 | 160 | 5.61 |
| json (structured) | 36.7 | 57 | 3.76 |
| code (repetitive) | 34.8 | 22 | 3.32 |
| quote (verbatim) | 31.5 | 54 | 3.43 |
| **mean** | **35.2** | | |

SPEC_DONE
