"""Execute the reproduction notebook on CPU and fail if any cell fails or a metric mismatches."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks/reproduce_results.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    print("notebook executed without errors")


if __name__ == "__main__":
    main()
