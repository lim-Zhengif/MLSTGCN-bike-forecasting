"""Seed-0 B1 accuracy screen; efficiency is a separate required measurement."""
import argparse
import json
import pandas as pd
import analyze_g0_sparse_gate as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-predictions", default=str(common.DEFAULT_B0))
    parser.add_argument("--b1-predictions", default=str(common.RESULT_ROOT /
        "holdout_b1_seed0_20260301_20260630_oracle" /
        "b1_seed0_20260301_20260630_oracle_station_hour_predictions.csv"))
    parser.add_argument("--output-dir", default=str(common.RESULT_ROOT / "b1_screen_seed0"))
    args = parser.parse_args()
    b0_path = common.absolute_path(args.b0_predictions)
    b1_path = common.absolute_path(args.b1_predictions)
    output = common.absolute_path(args.output_dir)
    b0 = common.load_predictions(b0_path, "B0")
    b1 = common.load_predictions(b1_path, "B1")
    audit = common.audit_comparability(b0, b1, 439200, 976)
    for path, expected in [(b0_path, "stgformer_official_core_bike_b0"),
                           (b1_path, "stgformer_official_static_multirelation_b1")]:
        summary = json.loads(path.with_name(path.name.replace(
            "_station_hour_predictions.csv", "_holdout_summary.json")).read_text(encoding="utf-8"))
        if summary["model"] != expected or summary["future_weather_regime"] != "oracle_observed_same_day":
            raise ValueError("Wrong model/weather regime for B1 screen")
        if summary.get("seed", 0) != 0:
            raise ValueError("This screen requires seed0")
    tables = {}
    for name, keys in [("overall", []), ("monthly", ["month"]), ("horizon", ["horizon"]),
                       ("anchor", ["anchor_hour"]), ("anchor_horizon", ["anchor_hour", "horizon"])]:
        tables[name] = pd.concat([common.grouped_metrics(f, keys) for f in [b0, b1]])
        common.atomic_csv(tables[name], output / (name + "_metrics.csv"))
    old, new = common.metric_record(b0), common.metric_record(b1)
    month = tables["monthly"].pivot(index="month", columns="model", values="mae")
    improvement = common.improvement_pct(old["mae"], new["mae"])
    conditions = dict(mae_improvement_at_least_1pct=improvement >= 1.0,
        rmse_net_not_both_degrade_over_1pct=not (
            common.degradation_pct(old["rmse"], new["rmse"]) > 1 and
            common.degradation_pct(old["mae_net"], new["mae_net"]) > 1),
        may_june_not_both_worse=not bool((month.loc[["2026-05", "2026-06"], "B1"] >
                                        month.loc[["2026-05", "2026-06"], "B0"]).all()))
    result = dict(audit=audit, b0=old, b1=new, improvement_pct=improvement,
                  accuracy_conditions=conditions, accuracy_screen_passed=bool(all(conditions.values())),
                  efficiency_status="PENDING_SEPARATE_BENCHMARK",
                  paired_days=common.paired_daily_test(b0, b1, 10000),
                  input_sha256={"b0": common.sha256_file(b0_path), "b1": common.sha256_file(b1_path)})
    common.atomic_json(result, output / "screen_summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
