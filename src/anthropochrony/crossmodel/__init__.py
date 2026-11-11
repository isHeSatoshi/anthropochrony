"""Exploratory cross-model arm: the same method on Aya Expanse 8B (not pre-registered).

New code only. Nothing here changes the pinned tiny-aya pipeline or its fingerprints.
"""

MODEL_ID = "CohereLabs/aya-expanse-8b"
MODEL_SHA = "5062468bf9bc0c6035fd64e06274333ec127d980"
EOT = 255001
BASE_PATH = "/out/aya-expanse-8b"
ADAPTER_RUN = "xm8b-s0"
HUB_ADAPTER = "isHeSatoshi/anthropochrony-aya-expanse-8b-lora-xm8b-s0"
TRAIN_TASKS = {"ALGO_FN": 70, "ALGO_IO": 50, "FUNC": 110, "LIB": 100}
K = 4
EVAL_CARDS = ("REF", "seen-cpu", "extrap-low", "heldout-190")
