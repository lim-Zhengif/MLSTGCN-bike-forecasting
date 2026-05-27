import json
import os
import runpy
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_ENTRY_SCRIPT_ENV = 'MLSTGCN_TRAIN_ENTRY_SCRIPT'
TRAIN_ENTRY_ARGS_ENV = 'MLSTGCN_TRAIN_ENTRY_ARGS_JSON'


def main():
    argv = sys.argv[1:]
    passthrough = list(argv)

    # This wrapper is a reproducible preset for the current best-performing
    # hourly training configuration: exp06_lr5e5_huber2_topk20.
    #
    # Any extra CLI arguments are still forwarded to train_bike.py unchanged,
    # so you can use this preset as a starting point and override only the
    # pieces you want to change, e.g.:
    #   python train_bike_hourly_exp06.py --project my_new_run
    #   python train_bike_hourly_exp06.py --graph_attention false --project exp08
    defaults = {
        '--data_dir': os.path.join('data', 'temporal_data', 'bike_hourly_safe_inventory'),
        '--graph_dir': os.path.join('data', 'graph', 'bike_hourly_safe_inventory'),
        '--project': 'bike_hourly_safe_inventory_exp06_lr5e5_huber2_topk20',
        '--pred_len': '24',
        '--hist_len': str(24 * 7),
        '--batch_size': '8',
        '--device': 'auto',
        '--logger': 'auto',
        '--epochs': '60',
        '--lr': '5e-5',
        '--weight_decay': '1e-4',
        '--loss': 'huber',
        '--huber_delta': '2.0',
        '--early_stop_patience': '8',
        '--lr_patience': '3',
        '--lr_factor': '0.5',
        '--min_lr': '1e-6',
        '--graph_sparsify_mode': 'topk',
        '--graph_topk': '20',
    }

    existing_flags = {item for item in passthrough if item.startswith('--')}
    final_argv = ['train_bike.py']
    for flag, value in defaults.items():
        if flag not in existing_flags:
            final_argv.extend([flag, value])
    final_argv.extend(passthrough)

    original_argv = sys.argv[:]
    original_entry_script = os.environ.get(TRAIN_ENTRY_SCRIPT_ENV)
    original_entry_args = os.environ.get(TRAIN_ENTRY_ARGS_ENV)
    try:
        os.environ[TRAIN_ENTRY_SCRIPT_ENV] = sys.argv[0]
        os.environ[TRAIN_ENTRY_ARGS_ENV] = json.dumps(sys.argv[1:], ensure_ascii=False)
        sys.argv = final_argv
        runpy.run_path(os.path.join(SCRIPT_DIR, 'train_bike.py'), run_name='__main__')
    finally:
        sys.argv = original_argv
        if original_entry_script is None:
            os.environ.pop(TRAIN_ENTRY_SCRIPT_ENV, None)
        else:
            os.environ[TRAIN_ENTRY_SCRIPT_ENV] = original_entry_script
        if original_entry_args is None:
            os.environ.pop(TRAIN_ENTRY_ARGS_ENV, None)
        else:
            os.environ[TRAIN_ENTRY_ARGS_ENV] = original_entry_args


if __name__ == '__main__':
    main()
