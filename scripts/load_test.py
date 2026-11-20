"""Send many requests to the API at the same time and report latency & throughput.

Usage:
    python scripts/load_test.py --url http://127.0.0.1:8000 --requests 200 --concurrency 20
"""
import argparse
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

SAMPLE_COMPLAINTS = [
    "There is a collection account on my credit report that does not belong to me.",
    "My mortgage servicer keeps charging me escrow fees I never agreed to.",
    "I was charged a late fee on my credit card even though I paid on time.",
    "A debt collector calls me every day about a debt I already paid off.",
    "My student loan payments were applied to the wrong loan.",
    "I sent money through a wire transfer and the recipient never got it.",
]


def send_request(url, text):
    """POST one complaint; return (latency in seconds, pod that answered)."""
    start = time.perf_counter()
    response = requests.post(f"{url}/predict", json={"text": text}, timeout=120)
    response.raise_for_status()
    return time.perf_counter() - start, response.json()["served_by"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    texts = [SAMPLE_COMPLAINTS[i % len(SAMPLE_COMPLAINTS)] for i in range(args.requests)]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda t: send_request(args.url, t), texts))
    total = time.perf_counter() - start

    latencies = sorted(r[0] for r in results)
    print(f"Requests:     {args.requests} ({args.concurrency} at a time)")
    print(f"Total time:   {total:.1f}s")
    print(f"Throughput:   {args.requests / total:.1f} requests/s")
    print(f"Latency p50:  {statistics.median(latencies) * 1000:.0f} ms")
    print(f"Latency p95:  {latencies[int(0.95 * len(latencies)) - 1] * 1000:.0f} ms")
    print("Requests per pod:")
    for pod, count in Counter(r[1] for r in results).most_common():
        print(f"  {pod}: {count}")


if __name__ == "__main__":
    main()
