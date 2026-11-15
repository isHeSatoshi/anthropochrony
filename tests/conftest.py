"""The benchmark code is a folder of scripts; make its modules importable by name."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/metatime"))
