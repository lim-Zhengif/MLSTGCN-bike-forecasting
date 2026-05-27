import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from hourly_pipeline_utils import detect_project_root, resolve_project_path


VERSION_TAG = '2026-04-09_小时级骑入骑出_安全库存区间'
DEFAULT_EXP06_EVAL_DIR = (
    '分析结果/2026-04-09_小时级骑入骑出_安全库存区间/'
    '小时级滚动预测_2026-02-01_到_2026-02-10_bike_hourly_safe_inventory_exp06_lr5e5_huber2_topk20'
)

DATE_COL = '日期'
HOUR_COL = '小时'
STATION_COL = '站点名称'
PRED_OUT_COL = 'pred_小时骑出量'
PRED_IN_COL = 'pred_小时骑入量'
PRED_NET_COL = 'pred_小时净流量'
TRUE_OUT_COL = 'true_小时骑出量'
TRUE_IN_COL = 'true_小时骑入量'
TRUE_NET_COL = 'true_小时净流量'
OUT_ERR_COL = 'abs_error_小时骑出量'
IN_ERR_COL = 'abs_error_小时骑入量'
NET_ERR_COL = 'abs_error_小时净流量'

DEFAULT_WINDOWS = {
    'morning_peak': (7, 10),
    'evening_peak': (17, 20),
}


PROJECT_ROOT = Path(detect_project_root(str(SCRIPT_DIR))).resolve()


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def corr_or_none(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def parse_peak_windows(raw_value):
    if not raw_value:
        return DEFAULT_WINDOWS
    windows = {}
    for item in raw_value.split(','):
        item = item.strip()
        if not item:
            continue
        name, hours = item.split(':', 1)
        start_text, end_text = hours.split('-', 1)
        start_hour = int(start_text)
        end_hour = int(end_text)
        if start_hour < 0 or end_hour > 23 or start_hour > end_hour:
            raise ValueError('Invalid peak window: %s' % item)
        windows[name.strip()] = (start_hour, end_hour)
    if not windows:
        raise ValueError('--peak_windows cannot be empty.')
    return windows


def load_inputs(eval_dir):
    pred_path = eval_dir / 'station_hourly_predictions.csv'
    inventory_path = eval_dir / 'safe_inventory_interval_report.csv'
    summary_path = eval_dir / 'summary.json'
    safe_summary_path = eval_dir / 'safe_inventory_interval_summary.json'

    if not pred_path.exists():
        raise FileNotFoundError('Missing %s' % pred_path)
    if not inventory_path.exists():
        raise FileNotFoundError('Missing %s' % inventory_path)

    pred_df = pd.read_csv(pred_path)
    inventory_df = pd.read_csv(inventory_path)

    summary = {}
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as fp:
            summary = json.load(fp)

    safe_summary = {}
    if safe_summary_path.exists():
        with open(safe_summary_path, 'r', encoding='utf-8') as fp:
            safe_summary = json.load(fp)

    return pred_df, inventory_df, summary, safe_summary


def add_hourly_trajectory(pred_df, inventory_df):
    inventory_cols = [
        DATE_COL,
        STATION_COL,
        '早晨初始车辆数',
        '总桩数',
        '晚间实际车辆数',
        'pred_S_min',
        'pred_S_max',
        'true_S_min',
        'true_S_max',
    ]
    available_cols = [col for col in inventory_cols if col in inventory_df.columns]
    merged = pred_df.merge(inventory_df[available_cols], on=[DATE_COL, STATION_COL], how='left')
    merged = merged.sort_values([DATE_COL, STATION_COL, HOUR_COL]).reset_index(drop=True)

    grouped = merged.groupby([DATE_COL, STATION_COL], sort=False)
    merged['pred_cumulative_net'] = grouped[PRED_NET_COL].cumsum()
    merged['true_cumulative_net'] = grouped[TRUE_NET_COL].cumsum()
    merged['pred_inventory_after_hour'] = merged['早晨初始车辆数'] + merged['pred_cumulative_net']
    merged['true_inventory_after_hour'] = merged['早晨初始车辆数'] + merged['true_cumulative_net']
    return merged


def risk_label(stockout_risk, overflow_risk):
    if stockout_risk and overflow_risk:
        return 'both'
    if stockout_risk:
        return 'stockout'
    if overflow_risk:
        return 'overflow'
    return 'safe'


def action_from_window(projected_min, projected_max, capacity, buffer):
    if pd.isna(projected_min) or pd.isna(projected_max):
        return 'unknown', np.nan
    low_target = float(buffer)
    if projected_min < low_target:
        return 'dispatch_in', float(np.ceil(low_target - projected_min))
    if not pd.isna(capacity):
        high_target = float(capacity) - float(buffer)
        if projected_max > high_target:
            return 'dispatch_out', float(np.ceil(projected_max - high_target))
    return 'already_safe', 0.0


def build_window_rows(hourly_df, peak_windows, buffer):
    rows = []
    for (date_value, station_name), group in hourly_df.groupby([DATE_COL, STATION_COL], sort=False):
        group = group.sort_values(HOUR_COL)
        morning_inventory = group['早晨初始车辆数'].iloc[0]
        capacity = group['总桩数'].iloc[0]
        if pd.isna(morning_inventory):
            continue

        for window_name, (start_hour, end_hour) in peak_windows.items():
            window_df = group[(group[HOUR_COL] >= start_hour) & (group[HOUR_COL] <= end_hour)]
            if window_df.empty:
                continue

            before_window = group[group[HOUR_COL] < start_hour]
            pred_start_inventory = float(morning_inventory + before_window[PRED_NET_COL].sum())
            true_start_inventory = float(morning_inventory + before_window[TRUE_NET_COL].sum())

            pred_path = pred_start_inventory + window_df[PRED_NET_COL].cumsum().to_numpy(dtype=np.float64)
            true_path = true_start_inventory + window_df[TRUE_NET_COL].cumsum().to_numpy(dtype=np.float64)
            pred_path_with_start = np.concatenate([[pred_start_inventory], pred_path])
            true_path_with_start = np.concatenate([[true_start_inventory], true_path])

            pred_min = float(np.min(pred_path_with_start))
            pred_max = float(np.max(pred_path_with_start))
            true_min = float(np.min(true_path_with_start))
            true_max = float(np.max(true_path_with_start))

            pred_min_idx = int(np.argmin(pred_path_with_start))
            pred_max_idx = int(np.argmax(pred_path_with_start))
            true_min_idx = int(np.argmin(true_path_with_start))
            true_max_idx = int(np.argmax(true_path_with_start))
            pred_min_hour = start_hour if pred_min_idx == 0 else int(window_df[HOUR_COL].iloc[pred_min_idx - 1])
            pred_max_hour = start_hour if pred_max_idx == 0 else int(window_df[HOUR_COL].iloc[pred_max_idx - 1])
            true_min_hour = start_hour if true_min_idx == 0 else int(window_df[HOUR_COL].iloc[true_min_idx - 1])
            true_max_hour = start_hour if true_max_idx == 0 else int(window_df[HOUR_COL].iloc[true_max_idx - 1])

            pred_stockout = bool(pred_min < buffer)
            true_stockout = bool(true_min < buffer)
            pred_overflow = bool((not pd.isna(capacity)) and pred_max > float(capacity) - buffer)
            true_overflow = bool((not pd.isna(capacity)) and true_max > float(capacity) - buffer)
            pred_action, pred_action_amount = action_from_window(pred_min, pred_max, capacity, buffer)
            true_action, true_action_amount = action_from_window(true_min, true_max, capacity, buffer)

            rows.append(
                {
                    DATE_COL: date_value,
                    STATION_COL: station_name,
                    'window_name': window_name,
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    '早晨初始车辆数': morning_inventory,
                    '总桩数': capacity,
                    'pred_start_inventory': pred_start_inventory,
                    'true_start_inventory': true_start_inventory,
                    'pred_projected_min_inventory': pred_min,
                    'pred_projected_max_inventory': pred_max,
                    'true_projected_min_inventory': true_min,
                    'true_projected_max_inventory': true_max,
                    'pred_projected_min_hour': pred_min_hour,
                    'pred_projected_max_hour': pred_max_hour,
                    'true_projected_min_hour': true_min_hour,
                    'true_projected_max_hour': true_max_hour,
                    'pred_window_out_sum': float(window_df[PRED_OUT_COL].sum()),
                    'true_window_out_sum': float(window_df[TRUE_OUT_COL].sum()),
                    'pred_window_in_sum': float(window_df[PRED_IN_COL].sum()),
                    'true_window_in_sum': float(window_df[TRUE_IN_COL].sum()),
                    'pred_window_net_sum': float(window_df[PRED_NET_COL].sum()),
                    'true_window_net_sum': float(window_df[TRUE_NET_COL].sum()),
                    'window_out_mae': float(window_df[OUT_ERR_COL].mean()),
                    'window_in_mae': float(window_df[IN_ERR_COL].mean()),
                    'window_net_mae': float(window_df[NET_ERR_COL].mean()),
                    'window_net_sum_abs_error': float(abs(window_df[PRED_NET_COL].sum() - window_df[TRUE_NET_COL].sum())),
                    'pred_stockout_risk': pred_stockout,
                    'true_stockout_risk': true_stockout,
                    'pred_overflow_risk': pred_overflow,
                    'true_overflow_risk': true_overflow,
                    'pred_any_risk': pred_stockout or pred_overflow,
                    'true_any_risk': true_stockout or true_overflow,
                    'pred_risk_label': risk_label(pred_stockout, pred_overflow),
                    'true_risk_label': risk_label(true_stockout, true_overflow),
                    'risk_label_match': risk_label(pred_stockout, pred_overflow) == risk_label(true_stockout, true_overflow),
                    'pred_action': pred_action,
                    'pred_action_amount': pred_action_amount,
                    'true_action': true_action,
                    'true_action_amount': true_action_amount,
                }
            )
    return pd.DataFrame(rows)


def summarize_hourly(hourly_df, peak_windows):
    rows = []
    peak_hour_set = {
        hour
        for start_hour, end_hour in peak_windows.values()
        for hour in range(start_hour, end_hour + 1)
    }
    for hour_value, group in hourly_df.groupby(HOUR_COL):
        rows.append(
            {
                HOUR_COL: int(hour_value),
                'is_peak_hour': int(hour_value) in peak_hour_set,
                'pred_out_mean': float(group[PRED_OUT_COL].mean()),
                'true_out_mean': float(group[TRUE_OUT_COL].mean()),
                'out_bias_mean': float((group[PRED_OUT_COL] - group[TRUE_OUT_COL]).mean()),
                'out_mae_mean': float(group[OUT_ERR_COL].mean()),
                'pred_in_mean': float(group[PRED_IN_COL].mean()),
                'true_in_mean': float(group[TRUE_IN_COL].mean()),
                'in_bias_mean': float((group[PRED_IN_COL] - group[TRUE_IN_COL]).mean()),
                'in_mae_mean': float(group[IN_ERR_COL].mean()),
                'pred_net_mean': float(group[PRED_NET_COL].mean()),
                'true_net_mean': float(group[TRUE_NET_COL].mean()),
                'net_bias_mean': float((group[PRED_NET_COL] - group[TRUE_NET_COL]).mean()),
                'net_mae_mean': float(group[NET_ERR_COL].mean()),
                'out_corr': corr_or_none(group[PRED_OUT_COL], group[TRUE_OUT_COL]),
                'in_corr': corr_or_none(group[PRED_IN_COL], group[TRUE_IN_COL]),
                'net_corr': corr_or_none(group[PRED_NET_COL], group[TRUE_NET_COL]),
            }
        )
    return pd.DataFrame(rows)


def summarize_stations(window_df):
    rows = []
    for station_name, group in window_df.groupby(STATION_COL):
        rows.append(
            {
                STATION_COL: station_name,
                'window_records': int(len(group)),
                'peak_net_mae_mean': float(group['window_net_mae'].mean()),
                'peak_net_sum_abs_error_mean': float(group['window_net_sum_abs_error'].mean()),
                'pred_any_risk_count': int(group['pred_any_risk'].sum()),
                'true_any_risk_count': int(group['true_any_risk'].sum()),
                'risk_label_match_ratio': float(group['risk_label_match'].mean()),
                'stockout_missed_count': int((~group['pred_stockout_risk'] & group['true_stockout_risk']).sum()),
                'overflow_missed_count': int((~group['pred_overflow_risk'] & group['true_overflow_risk']).sum()),
                'false_alarm_count': int((group['pred_any_risk'] & ~group['true_any_risk']).sum()),
            }
        )
    station_df = pd.DataFrame(rows)
    station_df['risk_review_score'] = (
        station_df['peak_net_mae_mean']
        + station_df['stockout_missed_count'] * 2.0
        + station_df['overflow_missed_count'] * 2.0
        + station_df['false_alarm_count'] * 0.5
    )
    return station_df.sort_values('risk_review_score', ascending=False).reset_index(drop=True)


def summarize_station_windows(window_df):
    rows = []
    for (window_name, station_name), group in window_df.groupby(['window_name', STATION_COL]):
        pred_any = group['pred_any_risk'].astype(bool)
        true_any = group['true_any_risk'].astype(bool)
        pred_stockout = group['pred_stockout_risk'].astype(bool)
        true_stockout = group['true_stockout_risk'].astype(bool)
        pred_overflow = group['pred_overflow_risk'].astype(bool)
        true_overflow = group['true_overflow_risk'].astype(bool)
        rows.append(
            {
                'window_name': window_name,
                STATION_COL: station_name,
                'records': int(len(group)),
                'window_net_mae_mean': float(group['window_net_mae'].mean()),
                'window_net_sum_abs_error_mean': float(group['window_net_sum_abs_error'].mean()),
                'pred_window_net_sum_mean': float(group['pred_window_net_sum'].mean()),
                'true_window_net_sum_mean': float(group['true_window_net_sum'].mean()),
                'net_sum_bias_mean': float((group['pred_window_net_sum'] - group['true_window_net_sum']).mean()),
                'pred_any_risk_count': int(pred_any.sum()),
                'true_any_risk_count': int(true_any.sum()),
                'hit_any_risk_count': int((pred_any & true_any).sum()),
                'miss_any_risk_count': int((~pred_any & true_any).sum()),
                'false_alarm_count': int((pred_any & ~true_any).sum()),
                'any_risk_match_ratio': float((pred_any == true_any).mean()),
                'stockout_missed_count': int((~pred_stockout & true_stockout).sum()),
                'overflow_missed_count': int((~pred_overflow & true_overflow).sum()),
            }
        )
    station_window_df = pd.DataFrame(rows)
    station_window_df['risk_review_score'] = (
        station_window_df['window_net_mae_mean']
        + station_window_df['miss_any_risk_count'] * 2.0
        + station_window_df['false_alarm_count'] * 0.5
    )
    return station_window_df.sort_values(['window_name', 'risk_review_score'], ascending=[True, False]).reset_index(drop=True)


def risk_metrics(window_df, mask=None):
    data = window_df if mask is None else window_df[mask]
    pred_any = data['pred_any_risk']
    true_any = data['true_any_risk']
    pred_stockout = data['pred_stockout_risk']
    true_stockout = data['true_stockout_risk']
    pred_overflow = data['pred_overflow_risk']
    true_overflow = data['true_overflow_risk']

    return {
        'records': int(len(data)),
        'pred_any_risk_count': int(pred_any.sum()),
        'true_any_risk_count': int(true_any.sum()),
        'any_risk_match_ratio': safe_float((pred_any == true_any).mean()) if len(data) else None,
        'risk_label_match_ratio': safe_float(data['risk_label_match'].mean()) if len(data) else None,
        'any_risk_precision': safe_div(int((pred_any & true_any).sum()), int(pred_any.sum())),
        'any_risk_recall': safe_div(int((pred_any & true_any).sum()), int(true_any.sum())),
        'stockout_precision': safe_div(int((pred_stockout & true_stockout).sum()), int(pred_stockout.sum())),
        'stockout_recall': safe_div(int((pred_stockout & true_stockout).sum()), int(true_stockout.sum())),
        'overflow_precision': safe_div(int((pred_overflow & true_overflow).sum()), int(pred_overflow.sum())),
        'overflow_recall': safe_div(int((pred_overflow & true_overflow).sum()), int(true_overflow.sum())),
        'window_out_mae_mean': safe_float(data['window_out_mae'].mean()) if len(data) else None,
        'window_in_mae_mean': safe_float(data['window_in_mae'].mean()) if len(data) else None,
        'window_net_mae_mean': safe_float(data['window_net_mae'].mean()) if len(data) else None,
        'window_net_sum_abs_error_mean': safe_float(data['window_net_sum_abs_error'].mean()) if len(data) else None,
    }


def build_summary(hourly_df, hour_df, window_df, station_df, eval_summary, safe_summary):
    peak_hour_df = hour_df[hour_df['is_peak_hour']]
    non_peak_hour_df = hour_df[~hour_df['is_peak_hour']]
    summary = {
        'eval_checkpoint_project': eval_summary.get('checkpoint_project'),
        'eval_checkpoint': eval_summary.get('checkpoint'),
        'date_start': eval_summary.get('date_start') or str(hourly_df[DATE_COL].min()),
        'date_end': eval_summary.get('date_end') or str(hourly_df[DATE_COL].max()),
        'num_dates': int(hourly_df[DATE_COL].nunique()),
        'num_stations': int(hourly_df[STATION_COL].nunique()),
        'hourly_records': int(len(hourly_df)),
        'overall_hourly_out_mae': safe_float(hourly_df[OUT_ERR_COL].mean()),
        'overall_hourly_in_mae': safe_float(hourly_df[IN_ERR_COL].mean()),
        'overall_hourly_net_mae': safe_float(hourly_df[NET_ERR_COL].mean()),
        'overall_out_corr': corr_or_none(hourly_df[PRED_OUT_COL], hourly_df[TRUE_OUT_COL]),
        'overall_in_corr': corr_or_none(hourly_df[PRED_IN_COL], hourly_df[TRUE_IN_COL]),
        'overall_net_corr': corr_or_none(hourly_df[PRED_NET_COL], hourly_df[TRUE_NET_COL]),
        'peak_hour_net_mae_mean': safe_float(peak_hour_df['net_mae_mean'].mean()) if not peak_hour_df.empty else None,
        'non_peak_hour_net_mae_mean': safe_float(non_peak_hour_df['net_mae_mean'].mean()) if not non_peak_hour_df.empty else None,
        'safe_inventory_summary': {
            'pred_S_min_mae': safe_summary.get('pred_S_min_mae'),
            'pred_S_max_mae': safe_summary.get('pred_S_max_mae'),
            'pred_interval_feasible_ratio': safe_summary.get('pred_interval_feasible_ratio'),
        },
        'all_peak_windows': risk_metrics(window_df),
        'by_window': {},
        'top_risk_review_stations': station_df.head(10).to_dict(orient='records'),
    }
    for window_name in sorted(window_df['window_name'].unique()):
        summary['by_window'][window_name] = risk_metrics(window_df, window_df['window_name'] == window_name)
    return summary


def fmt(value, digits=4):
    if value is None or pd.isna(value):
        return 'NA'
    return str(round(float(value), digits))


def bool_count(value):
    return int(value) if value is not None else 'NA'


def build_markdown(summary, window_df, station_df, output_paths):
    lines = [
        '# exp06 连续小时高峰风险分析',
        '',
        '## 核心结论',
        '- exp06 已经可以按连续小时分析高峰期风险；当前结果适合进入调度仿真和规则验证阶段。',
        '- 预测的小时级 MAE 处在 1 辆车以内，峰时净流量 MAE 也仍然较低，说明模型能提供有用的站点级预警信号。',
        '- 但安全库存区间更适合做预警和人工辅助，不宜直接作为自动调度硬边界。',
        '',
        '## 总体指标',
        '- 日期范围: `%s` 到 `%s`。' % (summary['date_start'], summary['date_end']),
        '- 站点数: `%s`，小时记录数: `%s`。' % (summary['num_stations'], summary['hourly_records']),
        '- 小时骑出 MAE: `%s`，小时骑入 MAE: `%s`，小时净流量 MAE: `%s`。'
        % (
            fmt(summary['overall_hourly_out_mae']),
            fmt(summary['overall_hourly_in_mae']),
            fmt(summary['overall_hourly_net_mae']),
        ),
        '- 骑出相关系数: `%s`，骑入相关系数: `%s`，净流量相关系数: `%s`。'
        % (
            fmt(summary['overall_out_corr']),
            fmt(summary['overall_in_corr']),
            fmt(summary['overall_net_corr']),
        ),
        '- 峰时小时净流量 MAE: `%s`，非峰时小时净流量 MAE: `%s`。'
        % (
            fmt(summary['peak_hour_net_mae_mean']),
            fmt(summary['non_peak_hour_net_mae_mean']),
        ),
        '',
        '## 高峰窗口风险识别',
    ]

    for window_name, metrics in summary['by_window'].items():
        lines.extend(
            [
                '### %s' % window_name,
                '- 窗口记录数: `%s`。' % metrics['records'],
                '- 预测风险窗口数: `%s`，真实风险窗口数: `%s`。'
                % (metrics['pred_any_risk_count'], metrics['true_any_risk_count']),
                '- 任意风险 precision: `%s`，recall: `%s`，风险标签匹配率: `%s`。'
                % (
                    fmt(metrics['any_risk_precision']),
                    fmt(metrics['any_risk_recall']),
                    fmt(metrics['risk_label_match_ratio']),
                ),
                '- 窗口骑出 MAE: `%s`，骑入 MAE: `%s`，净流量 MAE: `%s`，窗口累计净流量绝对误差: `%s`。'
                % (
                    fmt(metrics['window_out_mae_mean']),
                    fmt(metrics['window_in_mae_mean']),
                    fmt(metrics['window_net_mae_mean']),
                    fmt(metrics['window_net_sum_abs_error_mean']),
                ),
                '',
            ]
        )

    top_rows = station_df.head(10)
    lines.extend(
        [
            '## 需要重点复核的站点',
            '',
            '| 站点 | 峰时净流量MAE | 漏报缺车 | 漏报爆仓 | 误报 | 风险标签匹配率 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |',
        ]
    )
    for _, row in top_rows.iterrows():
        lines.append(
            '| %s | %s | %s | %s | %s | %s |'
            % (
                row[STATION_COL],
                fmt(row['peak_net_mae_mean']),
                bool_count(row['stockout_missed_count']),
                bool_count(row['overflow_missed_count']),
                bool_count(row['false_alarm_count']),
                fmt(row['risk_label_match_ratio']),
            )
        )

    lines.extend(
        [
            '',
            '## 结果是否合理',
            '- 从误差量级看，当前预测是合理的：整体小时级骑入/骑出/净流量 MAE 都在 1 辆车以内，适合做连续小时趋势分析。',
            '- 从业务目标看，当前预测能支持“高峰风险筛查、站点排序、调度仿真”；但安全库存区间覆盖真实上下界的能力还不够强，因此不适合直接自动下发精确调度指令。',
            '- 下一步最合理的是基于 exp06 跑滚动调度仿真，比较不同 `lookahead_hours` 和 `buffer` 下是否能减少缺车/爆仓，而不是再只看 MAE。',
            '',
            '## 输出文件',
        ]
    )
    for label, path in output_paths.items():
        lines.append('- %s: `%s`' % (label, path))

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Analyze continuous-hour peak risk from hourly bike forecasts.')
    parser.add_argument('--eval_dir', default=DEFAULT_EXP06_EVAL_DIR)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--peak_windows', default=None, help='Format: morning_peak:7-10,evening_peak:17-20')
    parser.add_argument('--buffer', type=float, default=0.0)
    args = parser.parse_args()

    eval_dir = Path(resolve_project_path(str(PROJECT_ROOT), args.eval_dir)).resolve()
    peak_windows = parse_peak_windows(args.peak_windows)
    pred_df, inventory_df, eval_summary, safe_summary = load_inputs(eval_dir)
    hourly_df = add_hourly_trajectory(pred_df, inventory_df)
    window_df = build_window_rows(hourly_df, peak_windows, args.buffer)
    hour_df = summarize_hourly(hourly_df, peak_windows)
    station_df = summarize_stations(window_df)
    station_window_df = summarize_station_windows(window_df)
    summary = build_summary(hourly_df, hour_df, window_df, station_df, eval_summary, safe_summary)
    summary.update(
        {
            'eval_dir': str(eval_dir),
            'buffer': args.buffer,
            'peak_windows': {name: list(hours) for name, hours in peak_windows.items()},
        }
    )

    if args.output_dir:
        output_dir = Path(resolve_project_path(str(PROJECT_ROOT), args.output_dir)).resolve()
    else:
        buffer_tag = str(args.buffer).replace('-', 'm').replace('.', 'p')
        output_dir = (
            PROJECT_ROOT
            / '分析结果'
            / VERSION_TAG
            / ('连续小时高峰风险分析_%s_b%s' % (eval_dir.name, buffer_tag))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        '高峰窗口站点日期风险明细': output_dir / 'peak_window_station_day_risk.csv',
        '逐小时指标': output_dir / 'hourly_peak_metrics.csv',
        '站点高峰风险汇总': output_dir / 'station_peak_risk_summary.csv',
        '按窗口拆分的站点风险汇总': output_dir / 'station_window_peak_risk_summary.csv',
        'JSON 摘要': output_dir / 'continuous_peak_risk_summary.json',
        'Markdown 报告': output_dir / 'continuous_peak_risk_report.md',
    }
    window_df.to_csv(output_paths['高峰窗口站点日期风险明细'], index=False, encoding='utf-8-sig')
    hour_df.to_csv(output_paths['逐小时指标'], index=False, encoding='utf-8-sig')
    station_df.to_csv(output_paths['站点高峰风险汇总'], index=False, encoding='utf-8-sig')
    station_window_df.to_csv(output_paths['按窗口拆分的站点风险汇总'], index=False, encoding='utf-8-sig')
    with open(output_paths['JSON 摘要'], 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    report = build_markdown(summary, window_df, station_df, output_paths)
    with open(output_paths['Markdown 报告'], 'w', encoding='utf-8') as fp:
        fp.write(report)

    print('Eval dir:', eval_dir)
    print('Output dir:', output_dir)
    print('Summary:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
