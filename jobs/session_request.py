"""Trusted RPC shim. No model output is evaluated or executed."""

import argparse
import json
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-ready", type=float, default=0)
    args = parser.parse_args()
    request = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(request) > 1024 * 1024:
        raise ValueError("Single-stream request exceeds 1 MiB")
    json.loads(request)
    deadline = time.monotonic() + args.wait_ready
    while True:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(max(180, args.wait_ready))
        try:
            connection.connect("/tmp/anthropochrony-generation.sock")
            break
        except (FileNotFoundError, ConnectionRefusedError):
            connection.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    with connection, connection.makefile("rwb") as stream:
        stream.write(request.rstrip(b"\n") + b"\n")
        stream.flush()
        response = stream.readline(8 * 1024 * 1024 + 1)
        if not response or len(response) > 8 * 1024 * 1024:
            raise RuntimeError("Missing or oversized generation response")
        json.loads(response)
        sys.stdout.buffer.write(response)


if __name__ == "__main__":
    main()
