# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.13.0", "transformers==5.21.1", "trl==1.7.1", "peft==0.21.2",
#   "aligntune==0.1.32", "huggingface-hub==2.4.0", "PyYAML==6.0.3",
# ]
# ///
"""AlignTune SFT on mounted, pinned inputs; private adapters only."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    "/code/src" if Path("/code/src").exists() else str(Path(__file__).resolve().parents[1] / "src"),
)


def main():
    import torch
    import yaml

    from anthropochrony.artifacts import write_json
    from anthropochrony.train.sft import train

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push-repo")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if not torch.cuda.is_available() or "L40S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This research training job requires the approved HF L40S GPU")
    write_json(
        Path(args.output) / "hardware.json",
        {
            "cuda": True,
            "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "job_id": os.environ.get("HF_JOB_ID"),
        },
    )
    train(
        args.base,
        args.dataset,
        args.output,
        config,
        args.seed,
        dry_run=args.dry_run,
        push_repo=args.push_repo,
    )


if __name__ == "__main__":
    main()
