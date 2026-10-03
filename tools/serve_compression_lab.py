#!/usr/bin/env python3
"""Serve the Netron Compression Lab as a local classroom web app.

This script uses only the Python standard library. It is copied into the
student offline bundle as ``serve.py`` so Raspberry Pi students do not need
Node.js or npm.
"""

from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Serve browser modules with a consistent type and fresh classroom assets."""

    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "application/javascript"}

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


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
    server = ThreadingHTTPServer((args.host, args.port), NoCacheHTTPRequestHandler)

    print(f"Netron Compression Lab is running on port {server.server_port}.")
    print("Access it through the JupyterLab proxy URL.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
