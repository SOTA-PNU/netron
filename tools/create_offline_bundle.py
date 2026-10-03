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
- The app server uses Python 3 and the standard library only.
- The classroom environment must provide a running JupyterLab server with
  jupyter-server-proxy installed and enabled.
- The laptop browser must already be able to access JupyterLab on the
  Raspberry Pi.
- Node.js / npm are not required to run this bundle.
- After environment preparation, no Internet connection is required to
  serve the bundled app.

Start the app in JupyterLab Terminal
1. Open a terminal and move to the lab folder containing both the notebook
   and compression-lab-offline.
2. Run:

   cd compression-lab-offline
   python3 serve.py

3. Keep the terminal and server running during the Compression Lab steps.

Open the app through JupyterLab
1. Start with the JupyterLab address already open in the laptop browser.
2. Remove /lab and any following path or query, then append /proxy/8080/.

   Example:
   JupyterLab:      http://localhost:8888/lab
   Compression Lab: http://localhost:8888/proxy/8080/

3. Keep the host and port from your actual JupyterLab address. Port 8888 is
   only an example. If JupyterLab uses a base path such as /class/lab,
   keep it: /class/proxy/8080/.
4. Open the resulting address in a new browser tab. No separate SSH
   forwarding of port 8080 is required for this proxy access.

Load files and use Compression Lab
1. Download models/cifar_10_model.keras and images/cifar10_test.png to the
   laptop as described in notebook 3-3, section 3-1.
2. Use Open Model to select the downloaded .keras file.
3. Select the final Dense layer, then its kernel in Weight tensor.
   The default classroom model uses a 64 x 10 kernel.
   Bias is excluded from selection and retained in model predictions.
4. Use the Pruning and Quantization tabs to compare weight changes.
   Both tables show the current range and total weight count. Use Previous
   and Next to browse ten weights per page.
   Applied Threshold reports the last successful pruning application.
   Quantization reports bits per weight, the theoretical bit reduction,
   and mean absolute error over all weights in the selected kernel.
5. Select the downloaded PNG in Test image to compare Top-3 predictions.
   Check its filename and the preview resized to the model input.
   Top-3 is a floating-point preview with only the selected tensor changed;
   it is not the whole-model INT8 benchmark.
6. When finished, return to the server terminal and press Ctrl+C.

If port 8080 is already in use
- Run python3 serve.py --port 8081 instead.
- Change the proxy path to /proxy/8081/ as well.
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
    index.write_text(content, encoding="utf-8", newline="\n")


def add_student_files() -> None:
    shutil.copy2(ROOT / "tools" / "serve_compression_lab.py", OUTPUT / "serve.py")
    (OUTPUT / "README.txt").write_text(STUDENT_README, encoding="utf-8", newline="\n")


def main() -> None:
    copy_web_assets()
    stamp_metadata()
    add_student_files()
    print(f"Created offline bundle: {OUTPUT}")
    print("Student command: python3 serve.py")


if __name__ == "__main__":
    main()
