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

    # This wrapper only injects hourly-task defaults.
    # Any extra CLI arguments are forwarded to train_bike.py unchanged,
    # so new train_bike.py parameters can be used directly from
    # `python train_bike_hourly.py ...`.
    defaults = {
        '--data_dir': os.path.join('data', 'temporal_data', 'bike_hourly_safe_inventory'),
        '--graph_dir': os.path.join('data', 'graph', 'bike_hourly_safe_inventory'),
        '--project': 'bike_hourly_safe_inventory',
        '--pred_len': '24',
        '--hist_len': str(24 * 7),
        '--batch_size': '8',
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
