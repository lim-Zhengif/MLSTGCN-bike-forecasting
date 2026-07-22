import sys

from train_bike_hourly_exp49_stgformer_spatiotemporal_residual_pred3 import (
    main as run_exp49,
)


def main():
    argv = sys.argv[1:]
    defaults = {
        '--project': 'exp60_exp49_demand_bucket_w100_100_115_130_hist168_pred3_8anchors_seed0_bs16',
        '--wandb_run_name': 'exp60_exp49_demand_bucket_w100_100_115_130_hist168_pred3_8anchors_seed0_bs16',
        '--wandb_project': 'top150_rolling6h_model_compare',
        '--graph_attention': 'false',
        '--demand_bucket_weighting': 'true',
        '--demand_bucket_quantiles': '0.5,0.8,0.95',
        '--demand_bucket_weights': '1.0,1.0,1.15,1.3',
        '--net_flow_consistency_weight': '0.0',
    }
    existing_flags = {item for item in argv if item.startswith('--')}
    injected = []
    for flag, value in defaults.items():
        if flag not in existing_flags:
            injected.extend([flag, value])

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] + injected + argv
        run_exp49()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
