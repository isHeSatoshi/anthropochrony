import ast
import re
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from anthropochrony.timemodel import SpeedCard, completion_time


def input_features(task, families, privileged=False):
    """Static, non-executing prompt features.

    ``task_turn`` is absent from the *public* release for restricted-source tasks
    (license compliance), so the text-derived columns fall back to the released
    ``prompt_tokens`` / ``ref_tokens`` counts, which are always published. Without
    this fallback the released artifacts cannot rebuild the workbench at all.
    """
    text = task.get("task_turn")
    if text:
        try:
            tree = ast.parse(text)
            functions = sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.walk(tree))
        except SyntaxError:
            functions = len(re.findall(r"\b(?:def|class)\s+\w+", text))
        n_chars, n_lines = len(text), len(text.splitlines())
    else:
        # Redacted statement: keep the feature vector shape identical by using the
        # published token counts as the length proxies.
        functions = 0
        n_chars = n_lines = 4 * int(task.get("prompt_tokens", 0))
    values = [
        task["prompt_tokens"],
        n_chars,
        n_lines,
        functions,
        len(task.get("public_examples", {})),
    ]
    values += [int(task["family"] == family) for family in families]
    if privileged:
        values.append(task["ref_tokens"])
    return values


class ConstantBaseline:
    def fit(self, tasks, labels):
        training = {t["task_id"] for t in tasks if t["split"] == "train"}
        self.by_card, self.by_family = defaultdict(list), defaultdict(list)
        for row in labels:
            if row["task_id"] in training:
                self.by_card[row["card_id"]].append(row["T_med_s"])
                self.by_family[row["card_id"], row["family"]].append(row["T_med_s"])
        if not self.by_card:
            raise ValueError("Constant baselines require train labels")
        return self

    def predict(self, task, card, family=False):
        values = self.by_family.get((card.id, task["family"])) if family else None
        values = values or self.by_card.get(card.id)
        if not values:
            raise ValueError(f"No train labels for card {card.id}")
        return float(np.median(values))


class InputRegressor:
    def __init__(self, privileged=False):
        self.privileged = privileged

    def fit(self, tasks, traces):
        train = [t for t in tasks if t["split"] == "train"]
        self.families = sorted({t["family"] for t in train})
        grouped = defaultdict(list)
        for trace in traces:
            if not trace.get("harness_error"):
                grouped[trace["task_id"]].append(trace)
        x, targets, feedback = [], [], []
        for task in train:
            runs = grouped.get(task["task_id"])
            if not runs:
                raise ValueError("Missing measured training trajectories")
            x.append(input_features(task, self.families, self.privileged))
            targets.append(
                [
                    np.log1p(np.median([r["total_o"] for r in runs])),
                    np.median([r["n_attempts"] for r in runs]),
                    np.log1p(np.median([r["total_exec_s"] for r in runs])),
                    np.log1p(np.median([r["total_u"] for r in runs])),
                ]
            )
            feedback += [a["u_tokens"] for r in runs for a in r["attempts"][1:]]
        self.models = [
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=max(2, min(20, len(train) // 10)),
                random_state=20261109,
            ).fit(x, np.array(targets)[:, i])
            for i in range(4)
        ]
        self.train_ids = [task["task_id"] for task in train]
        self.training_targets = np.asarray(targets)
        return self

    def predict_values(self, task):
        x = [input_features(task, self.families, self.privileged)]
        return np.asarray([float(m.predict(x)[0]) for m in self.models])

    def quantities(self, task):
        values = self.predict_values(task)
        return {
            "o_tokens": max(1, np.expm1(values[0])),
            "attempts": float(np.clip(values[1], 1, 3)),
            "exec_s": max(0, np.expm1(values[2])),
            "u_tokens": max(0, np.expm1(values[3])),
        }

    def predict(self, task, card):
        q = self.quantities(task)
        return completion_time(q["u_tokens"], q["o_tokens"], q["exec_s"], card, calls=q["attempts"])


class TunedInputRegressor(InputRegressor):
    """HGB plus embedding ridge; all selection uses val REF, never test labels."""

    def fit(self, tasks, traces, *, embeddings, labels, alphas=(1, 10, 100), blends=(0, 0.5, 1)):
        super().fit(tasks, traces)
        self.embeddings = embeddings
        index = {task["task_id"]: task for task in tasks}
        validation = [task for task in tasks if task["split"] == "val"]
        truth = {
            row["task_id"]: row
            for row in labels
            if row["split"] == "val" and row["card_id"] == "REF"
        }
        if not validation or any(task["task_id"] not in truth for task in validation):
            raise ValueError("B2 tuning requires complete val REF labels")
        x_train = np.asarray([self.ridge_features(index[task_id]) for task_id in self.train_ids])
        x_val = np.asarray([self.ridge_features(task) for task in validation])
        hgb = np.column_stack(
            [
                model.predict(
                    [input_features(task, self.families, self.privileged) for task in validation]
                )
                for model in self.models
            ]
        )
        candidates = []
        best = None
        for alpha in alphas:
            ridge = Ridge(alpha=alpha).fit(x_train, self.training_targets)
            predictions = ridge.predict(x_val)
            for blend in blends:
                combined = (1 - blend) * hgb + blend * predictions
                predicted_s = [
                    completion_time(
                        max(0, np.expm1(values[3])),
                        max(1, np.expm1(values[0])),
                        max(0, np.expm1(values[2])),
                        SpeedCard("REF", truth[task["task_id"]]["S"], truth[task["task_id"]]["P"]),
                        calls=float(np.clip(values[1], 1, 3)),
                    )
                    for task, values in zip(validation, combined, strict=True)
                ]
                actual = [truth[task["task_id"]]["T_med_s"] for task in validation]
                score = float(np.median(np.abs(np.log10(np.asarray(predicted_s) / actual))))
                candidates.append({"alpha": alpha, "ridge_weight": blend, "val_mdae": score})
                if best is None or score < best:
                    best, self.ridge, self.ridge_weight = score, ridge, blend
        self.selection = {
            "candidates": candidates,
            "val_mdae": best,
            "ridge_weight": self.ridge_weight,
            "alpha": self.ridge.alpha,
        }
        return self

    def ridge_features(self, task):
        return np.concatenate(
            [input_features(task, self.families, self.privileged), self.embeddings[task["task_id"]]]
        )

    def predict_values(self, task):
        hgb = super().predict_values(task)
        ridge = self.ridge.predict([self.ridge_features(task)])[0]
        return (1 - self.ridge_weight) * hgb + self.ridge_weight * ridge


class EmbeddingRidge:
    """Ridge on externally supplied pinned bge-small embeddings or probe activations."""

    def fit(self, features, targets, alpha=1.0):
        self.model = Ridge(alpha=alpha).fit(features, targets)
        return self

    def predict(self, features):
        return self.model.predict(features)


def rescale(reference_estimate, reference_card, card):
    return reference_estimate * reference_card.S / card.S
