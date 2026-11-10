# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["huggingface-hub==2.4.0"]
# ///
"""Run on cpu-basic, not a billed GPU, to stage gated weights into a private bucket."""

import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    snapshot_download(
        "CohereLabs/tiny-aya-global",
        revision="af89d219b53ed9b13b8a4645f9c8028973510324",
        local_dir=args.output,
        ignore_patterns=["*.onnx", "*.gguf", "*.bin"],
    )
    print("Pinned base model staged.")


if __name__ == "__main__":
    main()
