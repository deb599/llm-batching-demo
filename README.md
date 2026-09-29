# batching_demo

A tiny, dependency-free simulation that shows why grouping (batching) LLM requests
is the biggest lever on cost and speed. It serves 100 fake requests two ways,
one at a time versus a continuous-batching scheduler, and prints throughput and
p95 latency for each.

Companion to the article "Why your AI costs so much".

## Run it

Python 3, nothing to install:

```bash
python3 batching_demo.py
```

Example output:

```text
100 requests, 40ms/step, max_batch=16, max_wait=10ms
------------------------------------------------------------------------------
naive (batch=1)      wall=63502ms  throughput= 1.6 req/s  p50= 654ms  p95= 986ms
dynamic batching     wall= 4560ms  throughput=21.9 req/s  p50=2176ms  p95=4277ms
------------------------------------------------------------------------------
Speed-up: 13.9x higher throughput with the same model and hardware.
```

## Tune the dials

Edit the constants at the top of `batching_demo.py`:

- `MAX_BATCH` — how many requests share one step. Bigger raises throughput and the tail wait.
- `MAX_WAIT_MS` — how long the scheduler waits to fill a batch. Bigger is cheaper, slower first reply.
- `STEP_MS` — cost of one model step, kept constant to show that a step costs about the same for 1 request or a full batch.

The takeaway: batching lifts throughput a lot, and the price is a longer wait for the slowest requests.
