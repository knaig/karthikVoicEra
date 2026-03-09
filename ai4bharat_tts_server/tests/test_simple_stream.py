#!/usr/bin/env python3
"""Send text to TTS server, print chunks as they arrive, show time taken.
Usage: python test_simple_stream.py [ws://localhost:8002/ws] ["your text"]"""

import json
import sys
import time

try:
    from websocket import create_connection
except ImportError:
    print("Install: pip install websocket-client")
    sys.exit(1)

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8002/ws"
TEXT = sys.argv[2] if len(sys.argv) > 2 else "नमस्ते नमस्ते नमस्ते नमस्ते नमस्ते नमस्ते नमस्ते नमस्ते नमस्ते"
DESCRIPTION = "Female voice"

try:
    ws = create_connection(URL)
except Exception as e:
    print("Connection failed:", e)
    if "404" in str(e):
        print("Start server: uvicorn websocket_server:app --host 0.0.0.0 --port 8002")
    sys.exit(1)

payload = {"text": TEXT, "description": DESCRIPTION}
t_start = time.perf_counter()
ws.send(json.dumps(payload))

chunk_num = 0
ttft_ms = None

while True:
    msg = ws.recv()
    t_now = time.perf_counter()
    d = json.loads(msg)

    if "error" in d:
        print("Error:", d["error"])
        break

    if ttft_ms is None:
        ttft_ms = (t_now - t_start) * 1000
        print(f"[TTFT: {ttft_ms:.0f} ms]")

    chunk_b64 = d.get("chunk")
    if chunk_b64:
        chunk_num += 1
        size = len(chunk_b64)  # base64 length
        print(f"  chunk {chunk_num}: {size} chars (base64)")

    if d.get("is_final"):
        total_ms = (t_now - t_start) * 1000
        print(f"\nDone. Total time: {total_ms:.0f} ms | Chunks: {chunk_num}")
        break

ws.close()
