# -*- coding: utf-8 -*-
"""Evaluate G0 with the frozen B0 holdout pipeline and distinct outputs."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULT_ROOT = PROJECT_ROOT / "分析结果" / "对比实验" / "STGformerOfficialBike"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_stgformer_official_bike as b0_evaluator  # noqa: E402
from models import GATE_MODEL_ID, build_sparse_gate_from_checkpoint  # noqa: E402


def argument_value(name, default):
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError("%s requires a value" % name)
    return sys.argv[index + 1]


def append_default(name, value):
    if name not in sys.argv:
        sys.argv.extend([name, str(value)])


def main():
    seed = int(argument_value("--seed", "0"))
    lag_days = int(argument_value("--future_weather_lag_days", "0"))
    start_date = argument_value("--start_date", "2026-03-01")
    end_date = argument_value("--end_date", "2026-06-30")
    start_tag = start_date[:7].replace("-", "")
    end_tag = end_date[:7].replace("-", "")
    regime_tag = "oracle" if lag_days == 0 else "lag%d" % lag_days
    append_default("--seed", seed)
    append_default(
        "--checkpoint",
        RESULT_ROOT / ("g0_b0_sparse_demand_gate_seed%d" % seed) /
        "best_b0_sparse_demand_gate_g0.pt",
    )
    append_default(
        "--output_dir",
        RESULT_ROOT / ("g0_holdout_%s_%s_seed%d_%s" % (
            start_tag, end_tag, seed, regime_tag
        )),
    )
    append_default(
        "--eval_tag",
        "g0_b0_sparse_gate_seed%d_%s_%s_%s" % (
            seed, start_tag, end_tag, regime_tag
        ),
    )
    b0_evaluator.MODEL_ID = GATE_MODEL_ID
    b0_evaluator.build_model_from_checkpoint = build_sparse_gate_from_checkpoint
    b0_evaluator.main()


if __name__ == "__main__":
    main()
