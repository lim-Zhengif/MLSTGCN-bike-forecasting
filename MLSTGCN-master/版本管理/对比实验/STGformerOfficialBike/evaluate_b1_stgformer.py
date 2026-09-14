"""B1 evaluation using the same complete holdout protocol as B0."""
import argparse
import sys
import evaluate_stgformer_official_bike as evaluator
from train_stgformer_official_bike import DEFAULT_RESULT_ROOT
from models.multi_relation_graph_propagate import B1_MODEL_ID, build_b1_from_checkpoint
from b1_graphs import load_relations


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_date", default="2026-03-01")
    parser.add_argument("--end_date", default="2026-06-30")
    parser.add_argument("--future_weather_lag_days", type=int, default=0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--eval_tag")
    args, _ = parser.parse_known_args()
    regime = "oracle" if args.future_weather_lag_days == 0 else "lag%d" % args.future_weather_lag_days
    tag = "b1_seed%d_%s_%s_%s" % (args.seed, args.start_date.replace("-", ""),
                                  args.end_date.replace("-", ""), regime)
    defaults = dict(
        checkpoint=DEFAULT_RESULT_ROOT / ("b1_static_multirelation_seed%d_bs16" % args.seed) / "best_stgformer_b1.pt",
        output_dir=DEFAULT_RESULT_ROOT / ("holdout_" + tag), eval_tag=tag)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            sys.argv.extend(["--" + name, str(value)])
    evaluator.MODEL_ID = B1_MODEL_ID
    evaluator.build_model_from_checkpoint = build_b1_from_checkpoint
    evaluator.audit_graph_contract = lambda directory, name, nodes: load_relations(directory, nodes)[2]
    original_check = evaluator.assert_checkpoint_contract
    def check(checkpoint, metadata, graph_audit):
        original_check(checkpoint, metadata, graph_audit)
        if checkpoint["model_config"]["relation_audit"] != graph_audit:
            raise RuntimeError("B1 relation files/provenance differ from training")
    evaluator.assert_checkpoint_contract = check
    evaluator.main()


if __name__ == "__main__":
    main()
