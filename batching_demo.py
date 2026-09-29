"""
Proof for the LLM serving question:
  "Design an API that lets a customer sample from large generative models,
   and optimize batched calls for efficient use."

Core insight: an LLM forward pass costs roughly the same whether it serves 1
request or a whole batch (it is dominated by loading the model weights and the
fixed GPU step), so throughput scales with batch size. The win is a SCHEDULER
that groups many in-flight requests into batches, bounded by:
  - max_batch_size (GPU memory limit)
  - max_wait_ms   (latency budget: avoid making request #1 wait forever)

This script simulates that and compares naive one-at-a-time serving vs a
dynamic (continuous) batching scheduler. No external deps.
"""
import asyncio, time, random

STEP_MS = 40          # cost of ONE model forward step, near-constant up to MAX_BATCH
MAX_BATCH = 16        # GPU can process up to 16 sequences per step
MAX_WAIT_MS = 10      # scheduler waits at most this long to fill a batch
TOKENS_MIN, TOKENS_MAX = 5, 25   # tokens each request wants to generate


async def model_step(batch_size):
    # One forward step over batch_size sequences. Cost ~ constant, not x batch_size.
    await asyncio.sleep(STEP_MS / 1000)


# ---------- Naive: serve each request alone ----------
async def serve_naive(requests):
    start = time.perf_counter()
    latencies = []
    for r in requests:
        t0 = time.perf_counter()
        for _ in range(r["tokens"]):
            await model_step(1)          # batch of 1 every step
        latencies.append((time.perf_counter() - t0) * 1000)
    wall = (time.perf_counter() - start) * 1000
    return wall, latencies


# ---------- Dynamic batching scheduler (continuous batching) ----------
class Scheduler:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.running = True

    async def submit(self, req):
        fut = asyncio.get_event_loop().create_future()
        req["remaining"] = req["tokens"]
        req["future"] = fut
        req["t0"] = time.perf_counter()
        await self.queue.put(req)
        return await fut

    async def run(self):
        active = []                       # sequences currently generating
        while self.running or active or not self.queue.empty():
            deadline = time.perf_counter() + MAX_WAIT_MS / 1000
            while len(active) < MAX_BATCH:
                timeout = deadline - time.perf_counter()
                if timeout <= 0 and active:
                    break
                try:
                    req = await asyncio.wait_for(self.queue.get(), timeout=max(timeout, 0.001))
                    active.append(req)
                except asyncio.TimeoutError:
                    if active:
                        break
                    if not self.running:
                        return
            if not active:
                continue
            # One forward step advances EVERY active sequence by one token
            await model_step(len(active))
            done = []
            for req in active:
                req["remaining"] -= 1
                if req["remaining"] <= 0:
                    req["future"].set_result((time.perf_counter() - req["t0"]) * 1000)
                    done.append(req)
            for d in done:
                active.remove(d)          # finished seqs leave, new ones join next step


async def serve_batched(requests):
    sched = Scheduler()
    runner = asyncio.create_task(sched.run())
    start = time.perf_counter()

    async def fire(r):
        await asyncio.sleep(random.uniform(0, 0.02))   # stagger arrivals
        return await sched.submit(r)

    latencies = await asyncio.gather(*[fire(r) for r in requests])
    wall = (time.perf_counter() - start) * 1000
    sched.running = False
    await runner
    return wall, list(latencies)


def summary(name, wall, lat, n):
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat) // 2]
    p95 = lat_sorted[int(len(lat) * 0.95) - 1]
    print(f"{name:<20} wall={wall:7.0f}ms  throughput={n/(wall/1000):6.1f} req/s  "
          f"p50={p50:6.0f}ms  p95={p95:6.0f}ms")


async def main():
    random.seed(7)
    N = 100
    reqs = [{"id": i, "tokens": random.randint(TOKENS_MIN, TOKENS_MAX)} for i in range(N)]
    reqs2 = [dict(r) for r in reqs]

    w1, l1 = await serve_naive(reqs)
    w2, l2 = await serve_batched(reqs2)

    print(f"\n{N} requests, {STEP_MS}ms/step, max_batch={MAX_BATCH}, max_wait={MAX_WAIT_MS}ms\n" + "-" * 78)
    summary("naive (batch=1)", w1, l1, N)
    summary("dynamic batching", w2, l2, N)
    print("-" * 78)
    print(f"Speed-up: {w1/w2:.1f}x higher throughput with the same model and hardware.\n")


asyncio.run(main())
