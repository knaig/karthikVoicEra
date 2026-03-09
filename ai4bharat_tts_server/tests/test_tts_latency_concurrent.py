#!/usr/bin/env python3
"""
Test the TTS WebSocket server with configurable concurrent requests.
Measures time-to-first-token (TTFT) and total time per request.

Usage:
  python test_tts_latency_concurrent.py                    # 1 request (default)
  python test_tts_latency_concurrent.py -n 2              # 2 concurrent requests
  python test_tts_latency_concurrent.py -n 5              # 5 concurrent requests
  python test_tts_latency_concurrent.py -n 3 --url ws://localhost:8002/ws
"""

import argparse
import json
import sys
import threading
import time
from statistics import mean
from typing import List, Tuple

try:
    from websocket import create_connection
except ImportError:
    print("Install: pip install websocket-client")
    sys.exit(1)


DEFAULT_URL = "ws://localhost:8002/ws"
DEFAULT_TEXT = "नमस्ते नमस्ते नमस्ते नमस्ते नमस्ते"
DEFAULT_DESCRIPTION = "Female voice"


def run_single_request(
    request_id: int,
    url: str,
    text: str,
    description: str,
    barrier: threading.Barrier,
) -> Tuple[int, float, float]:
    """
    Open a WebSocket, wait at barrier (so all requests start together),
    send one request, then measure TTFT and total time.
    Returns (request_id, ttft_ms, total_ms).
    """
    ws = create_connection(url)
    try:
        payload = {"text": text, "description": description}
        barrier.wait()  # All threads start sending at roughly the same time
        t_send = time.perf_counter()
        ws.send(json.dumps(payload))

        ttft_ms = None
        total_ms = None

        while True:
            msg = ws.recv()
            t_recv = time.perf_counter()
            d = json.loads(msg)

            if "error" in d:
                raise RuntimeError(d["error"])

            if ttft_ms is None:
                ttft_ms = (t_recv - t_send) * 1000
            if d.get("is_final"):
                total_ms = (t_recv - t_send) * 1000
                break

        return (request_id, ttft_ms, total_ms)
    finally:
        ws.close()


def main():
    parser = argparse.ArgumentParser(
        description="Send N concurrent TTS requests and report TTFT + total time."
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1,
        metavar="N",
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"WebSocket URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize (default: Hindi greeting)",
    )
    parser.add_argument(
        "--description",
        default=DEFAULT_DESCRIPTION,
        help="Voice description (default: Female voice)",
    )
    args = parser.parse_args()

    if args.num_requests < 1:
        print("num-requests must be >= 1", file=sys.stderr)
        sys.exit(1)

    barrier = threading.Barrier(args.num_requests)
    results: List[Tuple[int, float, float]] = []
    errors: List[Tuple[int, str]] = []

    def run_and_capture(i: int) -> None:
        try:
            r = run_single_request(i + 1, args.url, args.text, args.description, barrier)
            results.append(r)
        except Exception as e:
            errors.append((i + 1, str(e)))

    threads = [
        threading.Thread(target=run_and_capture, args=(i,))
        for i in range(args.num_requests)
    ]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_wall = (time.perf_counter() - t_start) * 1000

    if errors:
        for rid, err in errors:
            print(f"Request {rid} failed: {err}", file=sys.stderr)
        if not results:
            sys.exit(1)

    # Sort by request_id for stable output
    results.sort(key=lambda x: x[0])

    print(f"\nConcurrent requests: {args.num_requests}")
    print("-" * 50)
    print(f"{'Request':<10} {'TTFT (ms)':<12} {'Total (ms)':<12}")
    print("-" * 50)
    for rid, ttft, total in results:
        print(f"{rid:<10} {ttft:<12.2f} {total:<12.2f}")

    if results:
        ttfts = [r[1] for r in results]
        totals = [r[2] for r in results]
        print("-" * 50)
        print(
            f"{'Min':<10} {min(ttfts):<12.2f} {min(totals):<12.2f}"
        )
        print(
            f"{'Avg':<10} {mean(ttfts):<12.2f} {mean(totals):<12.2f}"
        )
        print(
            f"{'Max':<10} {max(ttfts):<12.2f} {max(totals):<12.2f}"
        )
        print(f"\nWall-clock time (all {args.num_requests} requests): {t_wall:.2f} ms")
    print()


if __name__ == "__main__":
    main()
