import json
import os
import runpy
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_VERSION_DIR = SCRIPT_DIR
TRAIN_ENTRY_SCRIPT_ENV = 'MLSTGCN_TRAIN_ENTRY_SCRIPT'
TRAIN_ENTRY_ARGS_ENV = 'MLSTGCN_TRAIN_ENTRY_ARGS_JSON'
TRAIN_RECORD_DIR_ENV = 'MLSTGCN_TRAIN_RECORD_DIR'
TRAIN_VERSION_TAG_ENV = 'MLSTGCN_VERSION_TAG_OVERRIDE'


def main():
    argv = sys.argv[1:]
    passthrough = list(argv)

    defaults = {
        '--data_dir': os.path.join(
            'data',
            'temporal_data',
            'bike_hourly_safe_inventory_top150_nyc_full_predlen3_anchors_00_03_06_09_12_15_18_21_train2025_hist168',
        ),
        '--graph_dir': os.path.join(
            'data',
            'graph',
            'bike_hourly_safe_inventory_top150_exp10_anchor_hour_od_graph_train2025_hist168_pred3_8anchors',
        ),
        '--project': 'exp38_frozen_exp10_ast_tcn_edge_bias_final_head_bias005_hist168_pred3_8anchors_seed0',
        '--pred_len': '3',
        '--hist_len': str(24 * 7),
        '--batch_size': '8',
        '--device': 'auto',
        '--logger': 'wandb',
        '--epochs': '80',
        '--lr': '2e-5',
        '--weight_decay': '1e-4',
        '--loss': 'huber',
        '--huber_delta': '2.0',
        '--early_stop_patience': '10',
        '--lr_patience': '3',
        '--lr_factor': '0.5',
        '--min_lr': '1e-6',
        '--graph_sparsify_mode': 'topk',
        '--graph_topk': '20',
        '--graph_attention': 'true',
        '--matrix_weight': 'true',
        '--time_kernel_size': '3',
        '--graph_use': 'dist,neighb,distri,tempp,func,od00,od03,od06,od09,od12,od15,od18,od21',
        '--init_checkpoint': os.path.join(
            'wandb',
            'run-20260528_170031-1ksx3xae',
            'files',
            'top150_rolling6h_model_compare',
            '1ksx3xae',
            'checkpoints',
            'best-epoch=57-val_mae_epoch=val_mae_epoch=3.8071.ckpt',
        ),
        '--freeze_non_ast_tcn': 'true',
        '--freeze_trainable_scope': 'ast_tcn_final_head',
        '--ast_tcn_residual': 'true',
        '--ast_tcn_hidden_dim': '32',
        '--ast_tcn_layers': '4',
        '--ast_tcn_kernel_size': '3',
        '--ast_tcn_dilation_base': '2',
        '--ast_tcn_heads': '4',
        '--ast_tcn_dropout': '0.1',
        '--ast_tcn_residual_init': '0.01',
        '--ast_tcn_bounded_alpha': 'true',
        '--ast_tcn_alpha_max': '0.1',
        '--ast_tcn_zero_init': 'true',
        '--ast_tcn_edge_bias': 'true',
        '--ast_tcn_edge_bias_init': '0.05',
        '--ast_tcn_edge_bias_eps': '1e-6',
        '--channel_attention': 'false',
        '--channel_attention_reduction': '4',
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
    original_record_dir = os.environ.get(TRAIN_RECORD_DIR_ENV)
    original_version_tag = os.environ.get(TRAIN_VERSION_TAG_ENV)
    try:
        os.environ[TRAIN_ENTRY_SCRIPT_ENV] = sys.argv[0]
        os.environ[TRAIN_ENTRY_ARGS_ENV] = json.dumps(sys.argv[1:], ensure_ascii=False)
        os.environ[TRAIN_RECORD_DIR_ENV] = SCRIPT_DIR
        os.environ[TRAIN_VERSION_TAG_ENV] = os.path.basename(SCRIPT_DIR)
        sys.argv = final_argv
        runpy.run_path(os.path.join(SOURCE_VERSION_DIR, 'train_bike.py'), run_name='__main__')
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
        if original_record_dir is None:
            os.environ.pop(TRAIN_RECORD_DIR_ENV, None)
        else:
            os.environ[TRAIN_RECORD_DIR_ENV] = original_record_dir
        if original_version_tag is None:
            os.environ.pop(TRAIN_VERSION_TAG_ENV, None)
        else:
            os.environ[TRAIN_VERSION_TAG_ENV] = original_version_tag


if __name__ == '__main__':
    main()
