### 4slots-swaful (slots=4 workers=4 -fa off --swa-full)
```
requests: 12  workers: 4  wall: 29.1s
throughput: 0.41 req/s  9.3 completion-tokens/s
status: {200: 12}   failed/empty: 0
latency: mean 9.65s  p50 9.33s  p90 13.27s  p99 18.83s  max 18.83s
tokens per request: mean 22.7
server health after soak: 200
```

### 4slots-default (slots=4 workers=4 -fa off)
```
requests: 12  workers: 4  wall: 24.3s
throughput: 0.49 req/s  9.3 completion-tokens/s
status: {200: 12}   failed/empty: 0
latency: mean 7.82s  p50 8.21s  p90 10.10s  p99 12.01s  max 12.01s
tokens per request: mean 18.9
server health after soak: 200
```

### 4slots-fa-on (slots=4 workers=4 -fa on --swa-full)
```
requests: 12  workers: 4  wall: 5.7s
throughput: 2.11 req/s  43.2 completion-tokens/s
status: {200: 12}   failed/empty: 0
latency: mean 1.83s  p50 1.80s  p90 2.60s  p99 2.91s  max 2.91s
tokens per request: mean 20.5
server health after soak: 200
```

### 1slot (slots=1 workers=1 -fa off --swa-full)
```
requests: 8  workers: 1  wall: 16.3s
throughput: 0.49 req/s  10.8 completion-tokens/s
status: {200: 8}   failed/empty: 0
latency: mean 2.04s  p50 1.95s  p90 2.34s  p99 2.46s  max 2.46s
tokens per request: mean 22.0
server health after soak: 200
```

PAR_DONE
