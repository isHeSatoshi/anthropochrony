"""Execute the benchmark notebook on CPU and fail if any cell fails or a number mismatches."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks/benchmark_head_to_head.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    print("notebook executed without errors")


if __name__ == "__main__":
    main()
