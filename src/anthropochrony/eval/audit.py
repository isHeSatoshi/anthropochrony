import json
import math
from pathlib import Path

import auditkit as ak
from auditkit.runspec import RunConfig

from anthropochrony.artifacts import digest, write_json
from anthropochrony.eval.errors import log_error
from anthropochrony.parse import parse


@ak.scorer(direction=ak.Direction.MINIMIZE, name="ac_abs_log10_err_v1")
def abs_error(sample, output):
    return min(3.0, abs(log_error(parse(output).point_s, sample.metadata["T_med_s"])))


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="ac_within2x_v1")
def within2x(sample, output):
    return float(abs(log_error(parse(output).point_s, sample.metadata["T_med_s"])) <= math.log10(2))


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="ac_parse_ok_v1")
def parse_ok(sample, output):
    return float(parse(output).point_s is not None)


@ak.scorer(direction=ak.Direction.MINIMIZE, name="ac_human_units_v1")
def human_units(sample, output):
    return float(parse(output).has_human_units)


def evaluate(prompts, generate, directory, *, model_sha, code_sha):
    fingerprint = digest(
        {"prompts": prompts, "model_sha": model_sha, "code_sha": code_sha, "callable_schema": 2}
    )
    path = Path(directory) / fingerprint[:20]
    path.mkdir(parents=True, exist_ok=True)
    samples = [
        ak.Sample(
            id=row["est_id"],
            input=json.dumps({"est_id": row["est_id"], "messages": row["messages"]}),
            target=str(row["T_med_s"]),
            metadata=row,
        )
        for row in prompts
    ]
    config = RunConfig(
        temperature=0.0,
        max_tokens=max((row.get("max_tokens", 512) for row in prompts), default=512),
        seed=0,
        extra={"model_sha": model_sha, "evalset_sha": digest(prompts), "code_sha": code_sha},
    )
    result = ak.evaluate(
        samples,
        model=generate,
        scorers=[abs_error, within2x, parse_ok, human_units],
        config=config,
    )
    write_json(
        path / "run.json",
        {
            "fingerprint": fingerprint,
            "model_sha": model_sha,
            "code_sha": code_sha,
            "predictions": [
                {"id": p.sample_id, "raw_output": p.raw_output} for p in result.predictions
            ],
        },
        config=config.extra,
    )
    return result, fingerprint
