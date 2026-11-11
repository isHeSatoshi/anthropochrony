"""Cross-model arm driver (exploratory, not pre-registered): Aya Expanse 8B."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=[
            "stage",
            "measure",
            "labels",
            "data",
            "train-dry",
            "train",
            "estimate-base",
            "estimate-tuned",
            "report",
        ],
    )
    step = parser.parse_args().step
    from anthropochrony.crossmodel import pipeline

    if step == "stage":
        pipeline.stage(ROOT)
    elif step == "measure":
        pipeline.measure(ROOT)
    elif step == "labels":
        pipeline.build_labels_and_priors(ROOT)
    elif step == "data":
        print(pipeline.build_training_data(ROOT))
    elif step == "train-dry":
        pipeline.train(ROOT, max_steps=2)
    elif step == "train":
        pipeline.train(ROOT)
    elif step == "estimate-base":
        pipeline.estimate(ROOT, "base")
    elif step == "estimate-tuned":
        pipeline.estimate(ROOT, "tuned")
    else:
        pipeline.report(ROOT)


if __name__ == "__main__":
    main()
