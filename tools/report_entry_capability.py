from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for path in (ROOT, SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_entry_capability import main


if __name__ == "__main__":
    main()
