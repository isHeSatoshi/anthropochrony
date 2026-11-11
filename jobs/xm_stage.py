# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["huggingface-hub==2.4.0"]
# ///
"""Run on cpu-basic, not a billed GPU, to stage the cross-model weights into the private bucket."""

import argparse

from huggingface_hub import snapshot_download

REPO = "CohereLabs/aya-expanse-8b"
REVISION = "5062468bf9bc0c6035fd64e06274333ec127d980"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=args.output,
        ignore_patterns=["*.onnx", "*.gguf", "*.bin", "*.pt", "original/*"],
    )
    print("Pinned cross-model weights staged.")


if __name__ == "__main__":
    main()
