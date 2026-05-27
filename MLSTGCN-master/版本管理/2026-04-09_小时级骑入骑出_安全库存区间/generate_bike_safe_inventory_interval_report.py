import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from hourly_pipeline_utils import (
    build_safe_inventory_bounds,
    compute_adjustment_to_interval,
    detect_project_root,
    load_daily_feature_table,
    resolve_project_path,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = detect_project_root(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from analysis_result_utils import build_and_save_analysis_registry, get_analysis_root, resolve_version_tag


def find_latest_eval_dir(project_root, version_tag=None):
    analysis_root = get_analysis_root(project_root)
    patterns = []
    if version_tag:
        patterns.append(os.path.join(analysis_root, version_tag, '**', 'station_hourly_predictions.csv'))
    patterns.append(os.path.join(analysis_root, '**', 'station_hourly_predictions.csv'))

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    candidates = [os.path.abspath(path) for path in candidates if os.path.isfile(path)]
    if not candidates:
        raise FileNotFoundError(
            'Cannot auto-detect eval_dir because no station_hourly_predictions.csv was found under %s'
            % analysis_root
        )

    candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
    return os.path.dirname(candidates[0])


def build_report(pred_df, daily_feature_df):
    inventory_cols = ['早晨初始车辆数', '总桩数', '晚间实际车辆数']
    available_cols = ['日期', '站点名称'] + [col for col in inventory_cols if col in daily_feature_df.columns]
    merged = pred_df.merge(
        daily_feature_df[available_cols].drop_duplicates(subset=['日期', '站点名称'], keep='last'),
        on=['日期', '站点名称'],
        how='left',
    )
    for col in inventory_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    report_rows = []
    for (date_value, station_name), group in merged.groupby(['日期', '站点名称']):
        group = group.sort_values('小时')
        pred_net = group['pred_小时净流量'].to_numpy(dtype=np.float64)
        true_net = group['true_小时净流量'].to_numpy(dtype=np.float64)
        capacity = group['总桩数'].iloc[0] if '总桩数' in group.columns else np.nan
        morning_inventory = group['早晨初始车辆数'].iloc[0] if '早晨初始车辆数' in group.columns else np.nan
        actual_evening_inventory = group['晚间实际车辆数'].iloc[0] if '晚间实际车辆数' in group.columns else np.nan

        pred_s_min, pred_s_max, pred_cum, pred_feasible = build_safe_inventory_bounds(pred_net, capacity)
        true_s_min, true_s_max, true_cum, true_feasible = build_safe_inventory_bounds(true_net, capacity)
        pred_adjustment, pred_action = compute_adjustment_to_interval(morning_inventory, pred_s_min, pred_s_max)
        true_adjustment, true_action = compute_adjustment_to_interval(morning_inventory, true_s_min, true_s_max)

        pred_end_inventory = np.nan
        true_end_inventory = np.nan
        if not pd.isna(morning_inventory):
            pred_end_inventory = float(morning_inventory + pred_net.sum())
            true_end_inventory = float(morning_inventory + true_net.sum())

        report_rows.append(
            {
                '日期': date_value,
                '站点名称': station_name,
                '早晨初始车辆数': morning_inventory,
                '总桩数': capacity,
                '晚间实际车辆数': actual_evening_inventory,
                'pred_S_min': pred_s_min,
                'pred_S_max': pred_s_max,
                'pred_interval_feasible': pred_feasible,
                'true_S_min': true_s_min,
                'true_S_max': true_s_max,
                'true_interval_feasible': true_feasible,
                'pred_dispatch_action': pred_action,
                'pred_dispatch_amount': pred_adjustment,
                'true_dispatch_action': true_action,
                'true_dispatch_amount': true_adjustment,
                'pred_end_inventory': pred_end_inventory,
                'true_end_inventory': true_end_inventory,
                'pred_end_inventory_abs_error': abs(pred_end_inventory - true_end_inventory)
                if not pd.isna(pred_end_inventory) and not pd.isna(true_end_inventory) else np.nan,
                'pred_S_min_abs_error': abs(pred_s_min - true_s_min),
                'pred_S_max_abs_error': abs(pred_s_max - true_s_max) if not pd.isna(pred_s_max) and not pd.isna(true_s_max) else np.nan,
                'pred_hourly_net_mae': float(np.mean(np.abs(pred_net - true_net))),
                'pred_interval_contains_true_min': bool(pred_s_min <= true_s_min),
                'pred_interval_contains_true_max': bool(pd.isna(pred_s_max) or pd.isna(true_s_max) or pred_s_max >= true_s_max),
                'pred_cumulative_min': float(np.min(np.concatenate([[0.0], pred_cum]))),
                'pred_cumulative_max': float(np.max(np.concatenate([[0.0], pred_cum]))),
                'true_cumulative_min': float(np.min(np.concatenate([[0.0], true_cum]))),
                'true_cumulative_max': float(np.max(np.concatenate([[0.0], true_cum]))),
            }
        )

    report_df = pd.DataFrame(report_rows)
    summary = {
        'num_rows': int(len(report_df)),
        'pred_S_min_mae': float(report_df['pred_S_min_abs_error'].mean()) if not report_df.empty else None,
        'pred_S_max_mae': float(report_df['pred_S_max_abs_error'].dropna().mean()) if report_df['pred_S_max_abs_error'].notna().any() else None,
        'pred_end_inventory_mae': float(report_df['pred_end_inventory_abs_error'].dropna().mean()) if report_df['pred_end_inventory_abs_error'].notna().any() else None,
        'pred_interval_feasible_ratio': float(report_df['pred_interval_feasible'].mean()) if not report_df.empty else None,
        'true_interval_feasible_ratio': float(report_df['true_interval_feasible'].mean()) if not report_df.empty else None,
    }
    return report_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_dir',
        default=None,
        help='Directory produced by evaluate_bike_hourly_date_range.py. If omitted, auto-detect the latest result directory.',
    )
    parser.add_argument('--source_dir', default='二月份数据处理')
    parser.add_argument('--daily_feature_file', default='ST_Master_Feature_Table.csv')
    parser.add_argument('--aux_temporal_files', default='FINAL_JC_BaseTable_with_POI.csv,CLEAN_JC_BaseTable_2026-02-01_to_2026-02-28.csv')
    parser.add_argument('--weather_file', default=None)
    parser.add_argument('--output_name', default='safe_inventory_interval_report.csv')
    args = parser.parse_args()

    args.source_dir = resolve_project_path(PROJECT_ROOT, args.source_dir)
    version_tag = resolve_version_tag(PROJECT_ROOT, None, [args.eval_dir, SCRIPT_DIR])
    if args.eval_dir:
        args.eval_dir = resolve_project_path(PROJECT_ROOT, args.eval_dir)
    else:
        args.eval_dir = find_latest_eval_dir(PROJECT_ROOT, version_tag=version_tag)
        print('Auto-detected eval_dir:', args.eval_dir)
    pred_path = os.path.join(args.eval_dir, 'station_hourly_predictions.csv')
    if not os.path.exists(pred_path):
        raise FileNotFoundError('Missing station_hourly_predictions.csv under %s' % args.eval_dir)
    eval_summary_path = os.path.join(args.eval_dir, 'summary.json')
    eval_summary = None
    if os.path.exists(eval_summary_path):
        with open(eval_summary_path, 'r', encoding='utf-8') as fp:
            eval_summary = json.load(fp)

    pred_df = pd.read_csv(pred_path)
    target_dates = sorted(pred_df['日期'].dropna().astype(str).unique())
    station_names = sorted(pred_df['站点名称'].dropna().astype(str).unique())

    daily_feature_df, daily_feature_path, aux_paths, weather_path, aux_merge_summary = load_daily_feature_table(
        source_dir=args.source_dir,
        station_names=station_names,
        all_dates=target_dates,
        daily_feature_file=args.daily_feature_file,
        aux_temporal_files=args.aux_temporal_files,
        weather_file=args.weather_file,
    )
    report_df, summary = build_report(pred_df, daily_feature_df)

    output_csv = os.path.join(args.eval_dir, args.output_name)
    output_json = os.path.join(args.eval_dir, 'safe_inventory_interval_summary.json')
    report_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    summary.update(
        {
            'daily_feature_file': daily_feature_path,
            'aux_temporal_files': aux_paths,
            'weather_file': weather_path,
            'aux_merge_summary': aux_merge_summary,
            'version_tag': version_tag,
            'experiment_tag': eval_summary.get('experiment_tag') if eval_summary else None,
            'checkpoint_project': eval_summary.get('checkpoint_project') if eval_summary else None,
            'checkpoint': eval_summary.get('checkpoint') if eval_summary else None,
        }
    )
    with open(output_json, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    try:
        build_and_save_analysis_registry(PROJECT_ROOT)
    except ModuleNotFoundError as exc:
        if exc.name != 'openpyxl':
            raise
        print('Skip analysis registry Excel export because openpyxl is not installed.')
    print('Saved safe inventory interval report to:', output_csv)
    print('Saved safe inventory summary to:', output_json)
    print('Summary:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
