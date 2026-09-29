#!/usr/bin/env python3
"""Serve the Netron Compression Lab as a local classroom web app.

This script uses only the Python standard library. It is copied into the
student offline bundle as ``serve.py`` so Raspberry Pi students do not need
Node.js or npm.
"""

from __future__ import annotations

import argparse
import os
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def local_ip() -> str | None:
    """Return a useful LAN address when one can be determined."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Netron Compression Lab locally.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--directory",
        default=None,
        help="Directory to serve. Defaults to the folder containing this script.",
    )
    args = parser.parse_args()

    root = Path(args.directory).resolve() if args.directory else Path(__file__).resolve().parent
    if not (root / "index.html").exists():
        raise SystemExit(f"index.html not found in {root}")

    os.chdir(root)
    handler = SimpleHTTPRequestHandler
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print("Netron Compression Lab is running.")
    print(f"Local:   http://127.0.0.1:{args.port}")
    address = local_ip()
    if address:
        print(f"Network: http://{address}:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
