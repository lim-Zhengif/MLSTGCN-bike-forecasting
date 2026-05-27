import argparse
import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def detect_project_root(start_dir):
    current = Path(start_dir).resolve()
    while True:
        if (
            (current / 'data').is_dir()
            and (current / 'models').is_dir()
            and (current / 'datasets').is_dir()
        ):
            return current
        if current.parent == current:
            return Path(start_dir).resolve()
        current = current.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = detect_project_root(SCRIPT_DIR)
ANALYSIS_ROOT = PROJECT_ROOT / '分析结果'
CURRENT_VERSION_TAG = SCRIPT_DIR.name


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_latest_result_dir(version_tag, marker_file, prefer_keyword=None):
    version_root = ANALYSIS_ROOT / version_tag
    if not version_root.is_dir():
        raise FileNotFoundError('Missing analysis version directory: %s' % version_root)

    candidates = [path for path in version_root.rglob(marker_file) if path.is_file()]
    if prefer_keyword:
        preferred = [path for path in candidates if prefer_keyword.lower() in str(path.parent).lower()]
        if preferred:
            candidates = preferred
    if not candidates:
        raise FileNotFoundError('No %s found under %s' % (marker_file, version_root))
    candidates = sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0].parent


def find_latest_wandb_run():
    candidate_roots = [PROJECT_ROOT / 'wandb', SCRIPT_DIR / 'wandb']
    runs = []
    seen = set()
    for wandb_root in candidate_roots:
        if not wandb_root.is_dir():
            continue
        for path in wandb_root.glob('offline-run-*'):
            if not path.is_dir():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            runs.append(path)
    if not runs:
        return None
    runs = sorted(runs, key=lambda item: item.stat().st_mtime, reverse=True)
    return runs[0]


def load_json_if_exists(path):
    if not path or not Path(path).is_file():
        return None
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def infer_project_name_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return None
    normalized = os.path.normpath(str(checkpoint_path))
    parts = normalized.split(os.sep)
    if 'logs' in parts:
        idx = parts.index('logs')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if 'files' in parts:
        idx = parts.index('files')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def sanitize_run_tag(tag_value):
    if not tag_value:
        return None
    safe_chars = []
    for char in str(tag_value):
        if char.isalnum() or char in ('-', '_'):
            safe_chars.append(char)
        elif char in ('.', '='):
            safe_chars.append('-')
    safe_value = ''.join(safe_chars).strip('-_')
    return safe_value or None


def infer_run_tag(current_summary_json, current_eval_dir):
    if current_summary_json:
        explicit_tag = sanitize_run_tag(current_summary_json.get('experiment_tag'))
        if explicit_tag:
            return explicit_tag
        project_name = infer_project_name_from_checkpoint(current_summary_json.get('checkpoint'))
        if project_name:
            return sanitize_run_tag(project_name)
    return sanitize_run_tag(Path(current_eval_dir).name)


def find_wandb_run_dir_from_path(path_value):
    if not path_value:
        return None
    path_obj = Path(path_value)
    for candidate in [path_obj] + list(path_obj.parents):
        if candidate.name.startswith('offline-run-') and candidate.is_dir():
            return candidate
    return None


def load_training_summary_for_run(run_tag):
    if not run_tag:
        return None
    summary_path = ANALYSIS_ROOT / CURRENT_VERSION_TAG / ('training_result_%s' % run_tag) / 'training_summary.json'
    return load_json_if_exists(summary_path)


def resolve_wandb_run_dir(current_summary_json=None, inventory_summary_json=None, training_summary_json=None):
    candidate_summaries = [
        current_summary_json or {},
        inventory_summary_json or {},
        training_summary_json or {},
    ]
    for summary_json in candidate_summaries:
        for key in ('checkpoint', 'best_checkpoint', 'last_checkpoint', 'log_dir'):
            run_dir = find_wandb_run_dir_from_path(summary_json.get(key))
            if run_dir is not None:
                return run_dir
    latest_training_record = load_json_if_exists(SCRIPT_DIR / 'latest_hourly_training_checkpoint.json') or {}
    for key in ('best_checkpoint', 'last_checkpoint', 'log_dir'):
        run_dir = find_wandb_run_dir_from_path(latest_training_record.get(key))
        if run_dir is not None:
            return run_dir
    return find_latest_wandb_run()


def load_wandb_summary(current_summary_json=None, inventory_summary_json=None, training_summary_json=None):
    run_dir = resolve_wandb_run_dir(
        current_summary_json=current_summary_json,
        inventory_summary_json=inventory_summary_json,
        training_summary_json=training_summary_json,
    )
    if run_dir is None:
        return None

    debug_log = run_dir / 'logs' / 'debug.log'
    checkpoints_dir = next((run_dir / 'files').glob('**/checkpoints'), None)
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})')
    config_pattern = re.compile(r'^config:\s*(\{.*\})$')

    start_time = None
    end_time = None
    config = None
    if debug_log.is_file():
        with open(debug_log, 'r', encoding='utf-8', errors='ignore') as fp:
            for raw_line in fp:
                line = raw_line.strip()
                match = timestamp_pattern.match(line)
                if match:
                    dt = datetime.strptime('%s.%s' % (match.group(1), match.group(2)), '%Y-%m-%d %H:%M:%S.%f')
                    if start_time is None:
                        start_time = dt
                    end_time = dt
                config_match = config_pattern.match(line)
                if config_match:
                    try:
                        config = ast.literal_eval(config_match.group(1))
                    except Exception:
                        config = None

    checkpoints = []
    if checkpoints_dir is not None and checkpoints_dir.is_dir():
        for ckpt in sorted(checkpoints_dir.glob('*.ckpt')):
            checkpoints.append(
                {
                    'name': ckpt.name,
                    'size_bytes': ckpt.stat().st_size,
                    'modified_at': datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat(timespec='seconds'),
                }
            )

    duration_minutes = None
    if start_time is not None and end_time is not None:
        duration_minutes = round((end_time - start_time).total_seconds() / 60.0, 2)

    return {
        'run_dir': str(run_dir),
        'start_time': start_time.isoformat(timespec='seconds') if start_time else None,
        'end_time': end_time.isoformat(timespec='seconds') if end_time else None,
        'duration_minutes': duration_minutes,
        'config': config,
        'checkpoints': checkpoints,
    }


def build_current_daily_from_hourly(current_hourly_df):
    date_col = current_hourly_df.columns[0]
    station_col = current_hourly_df.columns[2]
    pred_out_col = current_hourly_df.columns[3]
    pred_in_col = current_hourly_df.columns[4]
    true_out_col = current_hourly_df.columns[6]
    true_in_col = current_hourly_df.columns[7]

    current_daily = current_hourly_df.groupby([date_col, station_col], as_index=False).agg(
        {
            pred_out_col: 'sum',
            pred_in_col: 'sum',
            true_out_col: 'sum',
            true_in_col: 'sum',
        }
    )
    current_daily['current_daily_out_abs_error'] = (current_daily[pred_out_col] - current_daily[true_out_col]).abs()
    current_daily['current_daily_in_abs_error'] = (current_daily[pred_in_col] - current_daily[true_in_col]).abs()
    return current_daily.rename(columns={date_col: '日期', station_col: '站点名称'})


def build_comparison_outputs(current_hourly_df, previous_station_df, hourly_metrics_df, daily_metrics_df, inventory_df):
    current_daily = build_current_daily_from_hourly(current_hourly_df)
    previous_daily = previous_station_df.rename(
        columns={
            previous_station_df.columns[0]: '日期',
            previous_station_df.columns[1]: '站点名称',
            previous_station_df.columns[4]: 'previous_daily_out_abs_error',
            previous_station_df.columns[7]: 'previous_daily_in_abs_error',
        }
    )

    merged = previous_daily.merge(
        current_daily[['日期', '站点名称', 'current_daily_out_abs_error', 'current_daily_in_abs_error']],
        on=['日期', '站点名称'],
        how='inner',
    )
    merged['out_abs_error_improvement'] = merged['previous_daily_out_abs_error'] - merged['current_daily_out_abs_error']
    merged['in_abs_error_improvement'] = merged['previous_daily_in_abs_error'] - merged['current_daily_in_abs_error']
    merged['both_targets_avg_improvement'] = (
        merged['out_abs_error_improvement'] + merged['in_abs_error_improvement']
    ) / 2.0
    merged['out_improved'] = merged['out_abs_error_improvement'] > 0
    merged['in_improved'] = merged['in_abs_error_improvement'] > 0
    merged['both_improved'] = merged['out_improved'] & merged['in_improved']

    station_improvement_df = merged.groupby('站点名称', as_index=False).agg(
        {
            'previous_daily_out_abs_error': 'mean',
            'current_daily_out_abs_error': 'mean',
            'previous_daily_in_abs_error': 'mean',
            'current_daily_in_abs_error': 'mean',
            'out_abs_error_improvement': 'mean',
            'in_abs_error_improvement': 'mean',
            'both_targets_avg_improvement': 'mean',
            'out_improved': 'mean',
            'in_improved': 'mean',
            'both_improved': 'mean',
        }
    )
    station_improvement_df = station_improvement_df.rename(
        columns={
            'previous_daily_out_abs_error': '上一版_日总骑出量_mae',
            'current_daily_out_abs_error': '当前版_小时聚合日总骑出量_mae',
            'previous_daily_in_abs_error': '上一版_日总骑入量_mae',
            'current_daily_in_abs_error': '当前版_小时聚合日总骑入量_mae',
            'out_abs_error_improvement': '骑出_mae改善值',
            'in_abs_error_improvement': '骑入_mae改善值',
            'both_targets_avg_improvement': '双目标平均改善值',
            'out_improved': '骑出改善占比',
            'in_improved': '骑入改善占比',
            'both_improved': '双目标同时改善占比',
        }
    ).sort_values('双目标平均改善值', ascending=False)

    hourly_col = hourly_metrics_df.columns[0]
    hourly_out_col = hourly_metrics_df.columns[1]
    hourly_in_col = hourly_metrics_df.columns[2]

    overall_summary = {
        'current_daily_out_mae_from_hourly': round(current_daily['current_daily_out_abs_error'].mean(), 4),
        'current_daily_in_mae_from_hourly': round(current_daily['current_daily_in_abs_error'].mean(), 4),
        'previous_daily_out_mae': round(previous_daily['previous_daily_out_abs_error'].mean(), 4),
        'previous_daily_in_mae': round(previous_daily['previous_daily_in_abs_error'].mean(), 4),
        'out_mae_improvement_pct': round(
            (previous_daily['previous_daily_out_abs_error'].mean() - current_daily['current_daily_out_abs_error'].mean())
            / previous_daily['previous_daily_out_abs_error'].mean()
            * 100.0,
            2,
        ),
        'in_mae_improvement_pct': round(
            (previous_daily['previous_daily_in_abs_error'].mean() - current_daily['current_daily_in_abs_error'].mean())
            / previous_daily['previous_daily_in_abs_error'].mean()
            * 100.0,
            2,
        ),
        'station_count_out_improved': int((station_improvement_df['骑出_mae改善值'] > 0).sum()),
        'station_count_in_improved': int((station_improvement_df['骑入_mae改善值'] > 0).sum()),
        'station_count_both_improved': int(
            ((station_improvement_df['骑出_mae改善值'] > 0) & (station_improvement_df['骑入_mae改善值'] > 0)).sum()
        ),
        'station_day_out_improved_ratio': round(merged['out_improved'].mean(), 4),
        'station_day_in_improved_ratio': round(merged['in_improved'].mean(), 4),
        'station_day_both_improved_ratio': round(merged['both_improved'].mean(), 4),
        'peak_7_10_out_mae': round(hourly_metrics_df.loc[hourly_metrics_df[hourly_col].between(7, 10), hourly_out_col].mean(), 4),
        'peak_7_10_in_mae': round(hourly_metrics_df.loc[hourly_metrics_df[hourly_col].between(7, 10), hourly_in_col].mean(), 4),
        'peak_17_20_out_mae': round(hourly_metrics_df.loc[hourly_metrics_df[hourly_col].between(17, 20), hourly_out_col].mean(), 4),
        'peak_17_20_in_mae': round(hourly_metrics_df.loc[hourly_metrics_df[hourly_col].between(17, 20), hourly_in_col].mean(), 4),
        'inventory_pred_S_min_mae': round(inventory_df[inventory_df.columns[18]].mean(), 4),
        'inventory_pred_S_max_mae': round(inventory_df[inventory_df.columns[19]].dropna().mean(), 4),
        'inventory_end_mae': round(inventory_df[inventory_df.columns[17]].dropna().mean(), 4),
        'inventory_feasible_ratio': round(inventory_df[inventory_df.columns[7]].mean(), 4),
        'inventory_contain_true_min_ratio': round(inventory_df[inventory_df.columns[21]].mean(), 4),
        'inventory_contain_true_max_ratio': round(inventory_df[inventory_df.columns[22]].mean(), 4),
        'inventory_contain_both_ratio': round(
            (inventory_df[inventory_df.columns[21]] & inventory_df[inventory_df.columns[22]]).mean(),
            4,
        ),
    }

    interval_station_df = inventory_df.groupby(inventory_df.columns[1], as_index=False).agg(
        {
            inventory_df.columns[18]: 'mean',
            inventory_df.columns[19]: 'mean',
            inventory_df.columns[17]: 'mean',
            inventory_df.columns[20]: 'mean',
        }
    ).sort_values(inventory_df.columns[17], ascending=False)

    best_days_df = daily_metrics_df.sort_values(daily_metrics_df.columns[1]).head(5)
    worst_days_df = daily_metrics_df.sort_values(daily_metrics_df.columns[1], ascending=False).head(5)

    return overall_summary, station_improvement_df, interval_station_df, best_days_df, worst_days_df


def build_chronic_station_outputs(current_hourly_df, exceed_threshold, chronic_min_count):
    station_col = current_hourly_df.columns[2]
    out_error_col = current_hourly_df.columns[9]
    in_error_col = current_hourly_df.columns[10]
    net_error_col = current_hourly_df.columns[11]

    chronic_df = current_hourly_df.copy()
    chronic_df['骑出误差超过阈值'] = chronic_df[out_error_col] > exceed_threshold
    chronic_df['骑入误差超过阈值'] = chronic_df[in_error_col] > exceed_threshold
    chronic_df['任一目标误差超过阈值'] = chronic_df['骑出误差超过阈值'] | chronic_df['骑入误差超过阈值']
    chronic_df['双目标同时超过阈值'] = chronic_df['骑出误差超过阈值'] & chronic_df['骑入误差超过阈值']

    station_chronic_df = chronic_df.groupby(station_col, as_index=False).agg(
        {
            out_error_col: 'mean',
            in_error_col: 'mean',
            net_error_col: 'mean',
            '骑出误差超过阈值': 'sum',
            '骑入误差超过阈值': 'sum',
            '任一目标误差超过阈值': 'sum',
            '双目标同时超过阈值': 'sum',
        }
    ).rename(
        columns={
            out_error_col: '小时级骑出量_mae',
            in_error_col: '小时级骑入量_mae',
            net_error_col: '小时级净流量_mae',
        }
    )
    total_hours = current_hourly_df.groupby([current_hourly_df.columns[0], current_hourly_df.columns[1]]).ngroups
    station_chronic_df['任一目标>%.1f占比' % exceed_threshold] = station_chronic_df['任一目标误差超过阈值'] / float(total_hours)
    station_chronic_df['风险等级'] = '一般'
    station_chronic_df.loc[station_chronic_df['任一目标误差超过阈值'] >= chronic_min_count, '风险等级'] = '顽疾站点'
    station_chronic_df.loc[station_chronic_df['任一目标误差超过阈值'] >= max(chronic_min_count * 2, 20), '风险等级'] = '重度顽疾站点'

    chronic_only_df = station_chronic_df[station_chronic_df['任一目标误差超过阈值'] >= chronic_min_count].copy()
    chronic_only_df = chronic_only_df.sort_values(
        ['任一目标误差超过阈值', '小时级净流量_mae'],
        ascending=[False, False],
    ).reset_index(drop=True)
    return station_chronic_df.sort_values('任一目标误差超过阈值', ascending=False), chronic_only_df


def build_single_day_peak_outputs(current_hourly_df, inventory_df, single_date):
    date_col = current_hourly_df.columns[0]
    hour_col = current_hourly_df.columns[1]
    station_col = current_hourly_df.columns[2]
    pred_out_col = current_hourly_df.columns[3]
    pred_in_col = current_hourly_df.columns[4]
    pred_net_col = current_hourly_df.columns[5]
    true_out_col = current_hourly_df.columns[6]
    true_in_col = current_hourly_df.columns[7]
    true_net_col = current_hourly_df.columns[8]
    out_err_col = current_hourly_df.columns[9]
    in_err_col = current_hourly_df.columns[10]
    net_err_col = current_hourly_df.columns[11]

    inv_date_col = inventory_df.columns[0]
    inv_station_col = inventory_df.columns[1]
    morning_inventory_col = inventory_df.columns[2]
    capacity_col = inventory_df.columns[3]
    pred_smin_col = inventory_df.columns[5]
    pred_smax_col = inventory_df.columns[6]
    pred_dispatch_action_col = inventory_df.columns[11]
    pred_dispatch_amount_col = inventory_df.columns[12]
    true_dispatch_action_col = inventory_df.columns[13]
    pred_end_inventory_col = inventory_df.columns[15]

    single_hourly = current_hourly_df[current_hourly_df[date_col] == single_date].copy()
    single_inventory = inventory_df[inventory_df[inv_date_col] == single_date].copy()

    morning_df = single_hourly[single_hourly[hour_col].between(7, 10)].groupby(station_col, as_index=False).agg(
        {
            pred_out_col: 'sum',
            pred_in_col: 'sum',
            pred_net_col: 'sum',
            true_out_col: 'sum',
            true_in_col: 'sum',
            true_net_col: 'sum',
            out_err_col: 'mean',
            in_err_col: 'mean',
            net_err_col: 'mean',
        }
    )
    morning_df.columns = [
        '站点名称',
        'pred_早高峰骑出量',
        'pred_早高峰骑入量',
        'pred_早高峰净流量',
        'true_早高峰骑出量',
        'true_早高峰骑入量',
        'true_早高峰净流量',
        '早高峰骑出_mae',
        '早高峰骑入_mae',
        '早高峰净流量_mae',
    ]

    evening_df = single_hourly[single_hourly[hour_col].between(17, 20)].groupby(station_col, as_index=False).agg(
        {
            pred_out_col: 'sum',
            pred_in_col: 'sum',
            pred_net_col: 'sum',
            true_out_col: 'sum',
            true_in_col: 'sum',
            true_net_col: 'sum',
            out_err_col: 'mean',
            in_err_col: 'mean',
            net_err_col: 'mean',
        }
    )
    evening_df.columns = [
        '站点名称',
        'pred_晚高峰骑出量',
        'pred_晚高峰骑入量',
        'pred_晚高峰净流量',
        'true_晚高峰骑出量',
        'true_晚高峰骑入量',
        'true_晚高峰净流量',
        '晚高峰骑出_mae',
        '晚高峰骑入_mae',
        '晚高峰净流量_mae',
    ]

    peak_df = single_inventory.merge(morning_df, left_on=inv_station_col, right_on='站点名称', how='left')
    peak_df = peak_df.merge(evening_df, left_on=inv_station_col, right_on='站点名称', how='left', suffixes=('', '_晚高峰'))
    peak_df['dispatch_signal_match'] = peak_df[pred_dispatch_action_col] == peak_df[true_dispatch_action_col]
    peak_df['pred_结束库存占容量比例'] = peak_df[pred_end_inventory_col] / peak_df[capacity_col].replace(0, pd.NA)
    peak_df['早高峰缺车预警缺口'] = peak_df[pred_smin_col] - peak_df[morning_inventory_col]
    peak_df['早高峰预测消耗强度'] = -peak_df['pred_早高峰净流量']
    peak_df['晚高峰预测累积强度'] = peak_df['pred_晚高峰净流量']
    peak_df['峰时最大误差'] = peak_df[
        [
            '早高峰骑出_mae',
            '早高峰骑入_mae',
            '早高峰净流量_mae',
            '晚高峰骑出_mae',
            '晚高峰骑入_mae',
            '晚高峰净流量_mae',
        ]
    ].max(axis=1)

    morning_shortage_df = peak_df[
        (peak_df[pred_dispatch_action_col] == 'dispatch_in') | (peak_df['早高峰缺车预警缺口'] > 0)
    ].copy()
    morning_shortage_df = morning_shortage_df.sort_values(
        ['早高峰缺车预警缺口', '早高峰预测消耗强度'],
        ascending=[False, False],
    )

    evening_overflow_df = peak_df[
        (peak_df[pred_dispatch_action_col] == 'dispatch_out') | (peak_df['pred_结束库存占容量比例'] > 0.8)
    ].copy()
    evening_overflow_df = evening_overflow_df.sort_values(
        ['pred_结束库存占容量比例', '晚高峰预测累积强度'],
        ascending=[False, False],
    )

    peak_error_df = peak_df.sort_values('峰时最大误差', ascending=False).copy()

    peak_summary = {
        'single_date': single_date,
        'dispatch_signal_match_ratio': round(peak_df['dispatch_signal_match'].mean(), 4),
        'dispatch_in_count': int((peak_df[pred_dispatch_action_col] == 'dispatch_in').sum()),
        'dispatch_out_count': int((peak_df[pred_dispatch_action_col] == 'dispatch_out').sum()),
        'already_safe_count': int((peak_df[pred_dispatch_action_col] == 'already_safe').sum()),
        'unknown_count': int((peak_df[pred_dispatch_action_col] == 'unknown').sum()),
        'morning_shortage_station_count': int(len(morning_shortage_df)),
        'evening_overflow_watch_station_count': int(len(evening_overflow_df)),
    }

    return peak_df, morning_shortage_df, evening_overflow_df, peak_error_df, peak_summary


def build_markdown(
    comparison_summary,
    current_summary_json,
    previous_summary_json,
    station_improvement_df,
    chronic_only_df,
    peak_summary,
    morning_shortage_df,
    evening_overflow_df,
    peak_error_df,
    wandb_summary,
    current_eval_dir,
    previous_eval_dir,
    single_date,
):
    improved_top = station_improvement_df.head(10)
    worse_top = station_improvement_df.sort_values('双目标平均改善值').head(10)

    lines = [
        '# 小时级版本对比与峰时风险解读',
        '',
        '## 评估范围',
        '- 当前版本结果目录: `%s`' % current_eval_dir,
        '- 上一版对比目录: `%s`' % previous_eval_dir,
        '- 单日重点分析日期: `%s`' % single_date,
        '',
        '## 整体对比结论',
        '- 当前小时级模型聚合成日总量后，骑出 MAE 从 %.4f 降到 %.4f，改善 %.2f%%。'
        % (
            comparison_summary['previous_daily_out_mae'],
            comparison_summary['current_daily_out_mae_from_hourly'],
            comparison_summary['out_mae_improvement_pct'],
        ),
        '- 骑入 MAE 从 %.4f 降到 %.4f，改善 %.2f%%。'
        % (
            comparison_summary['previous_daily_in_mae'],
            comparison_summary['current_daily_in_mae_from_hourly'],
            comparison_summary['in_mae_improvement_pct'],
        ),
        '- 105 个站点里，骑出改善站点 %d 个，骑入改善站点 %d 个，双目标同时改善站点 %d 个。'
        % (
            comparison_summary['station_count_out_improved'],
            comparison_summary['station_count_in_improved'],
            comparison_summary['station_count_both_improved'],
        ),
        '- 站点-日期层面，骑出改善占比 %.2f%%，骑入改善占比 %.2f%%，双目标同时改善占比 %.2f%%。'
        % (
            comparison_summary['station_day_out_improved_ratio'] * 100.0,
            comparison_summary['station_day_in_improved_ratio'] * 100.0,
            comparison_summary['station_day_both_improved_ratio'] * 100.0,
        ),
        '',
        '## 峰时段表现',
        '- 07:00-10:00 平均小时 MAE: 骑出 %.4f，骑入 %.4f。'
        % (comparison_summary['peak_7_10_out_mae'], comparison_summary['peak_7_10_in_mae']),
        '- 17:00-20:00 平均小时 MAE: 骑出 %.4f，骑入 %.4f。'
        % (comparison_summary['peak_17_20_out_mae'], comparison_summary['peak_17_20_in_mae']),
        '- 安全库存区间可行率 %.2f%%，但同时覆盖真实 S_min/S_max 的比例只有 %.2f%%，更适合做预警而不是硬边界。'
        % (
            comparison_summary['inventory_feasible_ratio'] * 100.0,
            comparison_summary['inventory_contain_both_ratio'] * 100.0,
        ),
        '',
        '## 站点层面最明显改善',
    ]

    for _, row in improved_top.iterrows():
        lines.append(
            '- `%s`: 骑出改善 %.3f，骑入改善 %.3f，双目标平均改善 %.3f。'
            % (row['站点名称'], row['骑出_mae改善值'], row['骑入_mae改善值'], row['双目标平均改善值'])
        )

    lines.extend(['', '## 仍需重点关注的退步站点'])
    for _, row in worse_top.iterrows():
        lines.append(
            '- `%s`: 骑出改善 %.3f，骑入改善 %.3f，双目标平均改善 %.3f。'
            % (row['站点名称'], row['骑出_mae改善值'], row['骑入_mae改善值'], row['双目标平均改善值'])
        )

    lines.extend(
        [
            '',
            '## 2026-02-01 峰时风险',
            '- 调度动作与真实区间风险一致率 %.2f%%。'
            % (peak_summary['dispatch_signal_match_ratio'] * 100.0),
            '- 该日模型给出 `%d` 个早高峰补车预警，`%d` 个晚高峰溢出观察站点。'
            % (peak_summary['morning_shortage_station_count'], peak_summary['evening_overflow_watch_station_count']),
            '',
            '### 早高峰补车预警 Top',
        ]
    )
    for _, row in morning_shortage_df.head(10).iterrows():
        lines.append(
            '- `%s`: 早晨库存 %.1f，pred_S_min %.1f，预测 07-10 净流量 %.3f，建议动作 `%s %.1f`。'
            % (
                row['站点名称'],
                row[morning_shortage_df.columns[2]],
                row[morning_shortage_df.columns[5]],
                row['pred_早高峰净流量'],
                row[morning_shortage_df.columns[11]],
                row[morning_shortage_df.columns[12]],
            )
        )

    lines.extend(['', '### 晚高峰满桩观察 Top'])
    for _, row in evening_overflow_df.head(10).iterrows():
        lines.append(
            '- `%s`: 结束库存/容量比例 %.3f，预测 17-20 净流量 %.3f，当前动作 `%s`。'
            % (
                row['站点名称'],
                row['pred_结束库存占容量比例'],
                row['pred_晚高峰净流量'],
                row[evening_overflow_df.columns[11]],
            )
        )

    lines.extend(['', '### 峰时误差最高站点'])
    for _, row in peak_error_df.head(10).iterrows():
        lines.append(
            '- `%s`: 峰时最大误差 %.3f，早高峰净流量 MAE %.3f，晚高峰净流量 MAE %.3f。'
            % (row['站点名称'], row['峰时最大误差'], row['早高峰净流量_mae'], row['晚高峰净流量_mae'])
        )

    lines.extend(['', '## 顽疾站点'])
    if chronic_only_df.empty:
        lines.append('- 当前阈值下未识别到顽疾站点。')
    else:
        lines.append(
            '- 这里按“10 天窗口内任一目标小时误差 > 3 辆车的次数 >= 10 次”定义顽疾站点，共 %d 个。'
            % len(chronic_only_df)
        )
        for _, row in chronic_only_df.head(12).iterrows():
            lines.append(
                '- `%s`: 超 3 辆车次数 %d，小时级骑出/骑入/净流量 MAE = %.3f / %.3f / %.3f，风险等级 `%s`。'
                % (
                    row['站点名称'],
                    int(row['任一目标误差超过阈值']),
                    row['小时级骑出量_mae'],
                    row['小时级骑入量_mae'],
                    row['小时级净流量_mae'],
                    row['风险等级'],
                )
            )

    lines.extend(['', '## 训练日志摘要'])
    if wandb_summary is None:
        lines.append('- 未找到可用的 wandb 离线 run 日志。')
    else:
        lines.append('- run 目录: `%s`' % wandb_summary['run_dir'])
        lines.append(
            '- 训练时间: `%s` 到 `%s`，约 %.2f 分钟。'
            % (
                wandb_summary['start_time'],
                wandb_summary['end_time'],
                wandb_summary['duration_minutes'] if wandb_summary['duration_minutes'] is not None else 0.0,
            )
        )
        if wandb_summary.get('config'):
            train_conf = wandb_summary['config'].get('train', {})
            graph_conf = wandb_summary['config'].get('graph', {})
            data_conf = wandb_summary['config'].get('data', {})
            lines.append(
                '- 关键配置: hist_len=%s, pred_len=%s, batch_size=%s, lr=%s, loss=%s, topk=%s。'
                % (
                    data_conf.get('hist_len'),
                    data_conf.get('pred_len'),
                    train_conf.get('batch_size'),
                    train_conf.get('lr'),
                    train_conf.get('loss'),
                    graph_conf.get('sparsify_topk'),
                )
            )
        if wandb_summary.get('checkpoints'):
            for ckpt in wandb_summary['checkpoints']:
                lines.append('- checkpoint: `%s`, 大小 %.2f MB。' % (ckpt['name'], ckpt['size_bytes'] / 1024.0 / 1024.0))

    if current_summary_json:
        lines.extend(
            [
                '',
                '## 当前版本摘要引用',
                '- 当前 10 天小时级 summary: `overall_mae_mean=%.4f`, `hourly_out_mae_mean=%.4f`, `hourly_in_mae_mean=%.4f`。'
                % (
                    current_summary_json.get('overall_mae_mean', 0.0),
                    current_summary_json.get('hourly_out_mae_mean', 0.0),
                    current_summary_json.get('hourly_in_mae_mean', 0.0),
                ),
            ]
        )

    if previous_summary_json:
        lines.append(
            '- 上一版 10 天日级 summary: `overall_mae_mean=%.4f`, `日间骑出量_mae_mean=%.4f`, `日间骑入量_mae_mean=%.4f`。'
            % (
                previous_summary_json.get('overall_mae_mean', 0.0),
                previous_summary_json.get('日间骑出量_mae_mean', 0.0),
                previous_summary_json.get('日间骑入量_mae_mean', 0.0),
            )
        )

    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_eval_dir', default=None)
    parser.add_argument('--previous_eval_dir', default=None)
    parser.add_argument('--previous_version_tag', default='2026-04-07_优化版_鲁棒损失_图稀疏化_星期嵌入')
    parser.add_argument('--single_date', default='2026-02-01')
    parser.add_argument('--exceed_threshold', type=float, default=3.0)
    parser.add_argument('--chronic_min_count', type=int, default=10)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()

    current_eval_dir = Path(args.current_eval_dir) if args.current_eval_dir else find_latest_result_dir(
        CURRENT_VERSION_TAG,
        'station_hourly_predictions.csv',
    )
    previous_eval_dir = Path(args.previous_eval_dir) if args.previous_eval_dir else find_latest_result_dir(
        args.previous_version_tag,
        'station_predictions.csv',
        prefer_keyword='log1p',
    )
    current_summary = load_json_if_exists(current_eval_dir / 'summary.json') or {}
    run_tag = infer_run_tag(current_summary, current_eval_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        date_start = current_summary.get('date_start', args.single_date)
        date_end = current_summary.get('date_end', args.single_date)
        folder_name = 'hourly_model_review_%s_to_%s' % (date_start, date_end)
        if run_tag:
            folder_name = '%s_%s' % (folder_name, run_tag)
        output_dir = ANALYSIS_ROOT / CURRENT_VERSION_TAG / folder_name
    ensure_dir(output_dir)

    current_hourly_df = pd.read_csv(current_eval_dir / 'station_hourly_predictions.csv')
    current_daily_metrics_df = pd.read_csv(current_eval_dir / 'daily_metrics.csv')
    current_hourly_metrics_df = pd.read_csv(current_eval_dir / 'hourly_metrics.csv')
    inventory_df = pd.read_csv(current_eval_dir / 'safe_inventory_interval_report.csv')
    previous_station_df = pd.read_csv(previous_eval_dir / 'station_predictions.csv')

    current_summary_json = current_summary
    previous_summary_json = load_json_if_exists(previous_eval_dir / 'summary.json')
    inventory_summary_json = load_json_if_exists(current_eval_dir / 'safe_inventory_interval_summary.json')
    training_summary_json = load_training_summary_for_run(run_tag)
    wandb_summary = load_wandb_summary(
        current_summary_json=current_summary_json,
        inventory_summary_json=inventory_summary_json,
        training_summary_json=training_summary_json,
    )

    comparison_summary, station_improvement_df, interval_station_df, best_days_df, worst_days_df = build_comparison_outputs(
        current_hourly_df,
        previous_station_df,
        current_hourly_metrics_df,
        current_daily_metrics_df,
        inventory_df,
    )
    station_chronic_df, chronic_only_df = build_chronic_station_outputs(
        current_hourly_df,
        exceed_threshold=args.exceed_threshold,
        chronic_min_count=args.chronic_min_count,
    )
    peak_df, morning_shortage_df, evening_overflow_df, peak_error_df, peak_summary = build_single_day_peak_outputs(
        current_hourly_df,
        inventory_df,
        single_date=args.single_date,
    )

    summary_payload = {
        'current_eval_dir': str(current_eval_dir),
        'previous_eval_dir': str(previous_eval_dir),
        'experiment_tag': run_tag,
        'single_date': args.single_date,
        'comparison_summary': comparison_summary,
        'peak_summary': peak_summary,
        'chronic_station_count': int(len(chronic_only_df)),
        'current_summary_json': current_summary_json,
        'previous_summary_json': previous_summary_json,
        'inventory_summary_json': inventory_summary_json,
        'wandb_summary': wandb_summary,
    }

    markdown_text = build_markdown(
        comparison_summary=comparison_summary,
        current_summary_json=current_summary_json,
        previous_summary_json=previous_summary_json,
        station_improvement_df=station_improvement_df,
        chronic_only_df=chronic_only_df,
        peak_summary=peak_summary,
        morning_shortage_df=morning_shortage_df,
        evening_overflow_df=evening_overflow_df,
        peak_error_df=peak_error_df,
        wandb_summary=wandb_summary,
        current_eval_dir=current_eval_dir,
        previous_eval_dir=previous_eval_dir,
        single_date=args.single_date,
    )

    station_improvement_df.to_csv(output_dir / 'station_improvement_comparison.csv', index=False, encoding='utf-8-sig')
    interval_station_df.to_csv(output_dir / 'inventory_interval_error_by_station.csv', index=False, encoding='utf-8-sig')
    best_days_df.to_csv(output_dir / 'best_days.csv', index=False, encoding='utf-8-sig')
    worst_days_df.to_csv(output_dir / 'worst_days.csv', index=False, encoding='utf-8-sig')
    station_chronic_df.to_csv(output_dir / 'station_hourly_error_overview.csv', index=False, encoding='utf-8-sig')
    chronic_only_df.to_csv(output_dir / 'chronic_stations.csv', index=False, encoding='utf-8-sig')
    peak_df.to_csv(output_dir / ('peak_station_risk_%s.csv' % args.single_date), index=False, encoding='utf-8-sig')
    morning_shortage_df.to_csv(output_dir / ('morning_shortage_risk_%s.csv' % args.single_date), index=False, encoding='utf-8-sig')
    evening_overflow_df.to_csv(output_dir / ('evening_overflow_watch_%s.csv' % args.single_date), index=False, encoding='utf-8-sig')
    peak_error_df.to_csv(output_dir / ('peak_error_top_%s.csv' % args.single_date), index=False, encoding='utf-8-sig')
    with open(output_dir / 'overall_comparison_summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary_payload, fp, ensure_ascii=False, indent=2)
    with open(output_dir / 'hourly_model_review.md', 'w', encoding='utf-8') as fp:
        fp.write(markdown_text)
    if run_tag:
        with open(output_dir / ('overall_comparison_summary_%s.json' % run_tag), 'w', encoding='utf-8') as fp:
            json.dump(summary_payload, fp, ensure_ascii=False, indent=2)
        with open(output_dir / ('hourly_model_review_%s.md' % run_tag), 'w', encoding='utf-8') as fp:
            fp.write(markdown_text)
        with open(output_dir / ('对比结论_%s.md' % run_tag), 'w', encoding='utf-8') as fp:
            fp.write(markdown_text)

    print('Current eval dir:', current_eval_dir)
    print('Previous eval dir:', previous_eval_dir)
    print('Experiment tag:', run_tag if run_tag else 'None')
    print('Saved review outputs to:', output_dir)
    print('Comparison summary:', json.dumps(comparison_summary, ensure_ascii=False))
    print('Peak summary:', json.dumps(peak_summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
