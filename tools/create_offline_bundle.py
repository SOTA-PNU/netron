#!/usr/bin/env python3
"""Create a student-ready Netron Compression Lab bundle without npm.

The upstream Netron web build currently copies browser assets from ``source``
and removes desktop-only entry points. For the classroom fork we reproduce that
packaging step with Python so the instructor and students do not need Node.js.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
OUTPUT = ROOT / "dist" / "compression-lab-offline"

WEB_EXTENSIONS = {".html", ".css", ".js", ".json", ".ico", ".png"}
EXCLUDED_FILES = {"app.js", "node.js", "desktop.mjs"}


STUDENT_README = """Netron Compression Lab - Offline Classroom Bundle
=================================================

Requirements
- Python 3 only
- Node.js / npm are NOT required
- Internet connection is NOT required

Raspberry Pi / JupyterLab Terminal
1. Open a terminal in this folder.
2. Run:

   python3 serve.py

3. The terminal prints an address such as:

   http://192.168.0.10:8080

4. Open that address in the PC browser connected to the same network.
5. Open the .keras model and select a layer with weights.
6. Use the Compression Lab panel for Pruning and Quantization practice.

Stop the server with Ctrl+C.

If port 8080 is already in use:

   python3 serve.py --port 8081
"""


def copy_web_assets() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    for path in SOURCE.rglob("*"):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_FILES or path.suffix.lower() not in WEB_EXTENSIONS:
            continue
        relative = path.relative_to(SOURCE)
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def stamp_metadata() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    index = OUTPUT / "index.html"
    content = index.read_text(encoding="utf-8")
    content = re.sub(
        r'(<meta\s+name="version"\s+content=")[^"]*(">)',
        rf'\g<1>{package.get("version", "0.0.0")}\g<2>',
        content,
        count=1,
    )
    content = re.sub(
        r'(<meta\s+name="date"\s+content=")[^"]*(">)',
        rf'\g<1>{package.get("date", "")}\g<2>',
        content,
        count=1,
    )
    index.write_text(content, encoding="utf-8")


def add_student_files() -> None:
    shutil.copy2(ROOT / "tools" / "serve_compression_lab.py", OUTPUT / "serve.py")
    (OUTPUT / "README.txt").write_text(STUDENT_README, encoding="utf-8")


def main() -> None:
    copy_web_assets()
    stamp_metadata()
    add_student_files()
    print(f"Created offline bundle: {OUTPUT}")
    print("Student command: python3 serve.py")


if __name__ == "__main__":
    main()
