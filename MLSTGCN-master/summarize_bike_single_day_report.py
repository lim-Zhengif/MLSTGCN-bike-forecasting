import argparse
import json
import os

import numpy as np
import pandas as pd

from analysis_result_utils import (
    build_analysis_task_dir,
    build_and_save_analysis_registry,
    ensure_analysis_dir,
    resolve_version_tag,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR


def resolve_project_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def safe_mean(values):
    valid_values = [value for value in values if pd.notna(value)]
    if not valid_values:
        return np.nan
    return float(np.mean(valid_values))


def resolve_column(df: pd.DataFrame, candidates, label: str, required: bool = True):
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f'缺少 {label}，可选列: {candidates}')
    return None


def resolve_default_report_file(version_tag: str, target_date: str, extra_tag: str = None) -> str:
    report_dir = build_analysis_task_dir(
        project_root=PROJECT_ROOT,
        version_tag=version_tag,
        task_type='单日库存推演',
        start_date=target_date,
        end_date=target_date,
        extra_tag=extra_tag,
    )
    return os.path.join(report_dir, '站点级单日库存推演对比表.csv')


def build_summary_dataframe(report_df: pd.DataFrame) -> pd.DataFrame:
    station_col = resolve_column(report_df, ['站点名称', 'station_name', 'Station'], '站点名称')
    node_col = resolve_column(report_df, ['Node_ID'], 'Node_ID', required=False)

    pred_out_col = resolve_column(report_df, ['pred_日间骑出量'], 'pred_日间骑出量')
    true_out_col = resolve_column(report_df, ['true_日间骑出量', '日间骑出量'], 'true_日间骑出量')
    pred_in_col = resolve_column(report_df, ['pred_日间骑入量'], 'pred_日间骑入量')
    true_in_col = resolve_column(report_df, ['true_日间骑入量', '日间骑入量'], 'true_日间骑入量')
    pred_net_col = resolve_column(report_df, ['pred_日间纯用户净流量'], 'pred_日间纯用户净流量', required=False)
    true_net_col = resolve_column(report_df, ['true_日间纯用户净流量', '日间纯用户净流量'], 'true_日间纯用户净流量', required=False)

    clean_flag_col = resolve_column(report_df, ['是否有clean真实标签'], '是否有clean真实标签', required=False)
    intervention_col = resolve_column(report_df, ['官方白天干预量'], '官方白天干预量', required=False)

    report_df = report_df.copy()
    report_df['骑出量绝对误差'] = (report_df[pred_out_col] - report_df[true_out_col]).abs()
    report_df['骑入量绝对误差'] = (report_df[pred_in_col] - report_df[true_in_col]).abs()

    if pred_net_col and true_net_col:
        report_df['净流量绝对误差'] = (report_df[pred_net_col] - report_df[true_net_col]).abs()
    else:
        net_error_col = resolve_column(report_df, ['预测净流量误差'], '预测净流量误差', required=False)
        report_df['净流量绝对误差'] = report_df[net_error_col].abs() if net_error_col else np.nan

    report_df['流量综合误差'] = report_df.apply(
        lambda row: safe_mean(
            [
                row.get('骑出量绝对误差'),
                row.get('骑入量绝对误差'),
                row.get('净流量绝对误差'),
            ]
        ),
        axis=1,
    )
    report_df['库存综合误差'] = report_df.apply(
        lambda row: safe_mean(
            [
                abs(row['预测库存_vs_真实理论误差']) if '预测库存_vs_真实理论误差' in report_df.columns and pd.notna(row['预测库存_vs_真实理论误差']) else np.nan,
                abs(row['预测库存_vs_真实实际误差']) if '预测库存_vs_真实实际误差' in report_df.columns and pd.notna(row['预测库存_vs_真实实际误差']) else np.nan,
                abs(row['预测库存_vs_晚间快照误差']) if '预测库存_vs_晚间快照误差' in report_df.columns and pd.notna(row['预测库存_vs_晚间快照误差']) else np.nan,
            ]
        ),
        axis=1,
    )
    report_df['总综合误差'] = report_df.apply(
        lambda row: safe_mean([row.get('流量综合误差'), row.get('库存综合误差')]),
        axis=1,
    )

    if clean_flag_col:
        summary_df = report_df[report_df[clean_flag_col] == True].copy()
        if summary_df.empty:
            summary_df = report_df.copy()
    else:
        summary_df = report_df.copy()

    summary_df = summary_df.sort_values(
        ['总综合误差', '库存综合误差', '流量综合误差', station_col]
    ).reset_index(drop=True)
    summary_df['综合表现排名'] = np.arange(1, len(summary_df) + 1)

    output_columns = ['综合表现排名']
    if node_col:
        output_columns.append(node_col)
    output_columns.extend(
        [
            station_col,
            '骑出量绝对误差',
            '骑入量绝对误差',
            '净流量绝对误差',
            '流量综合误差',
        ]
    )
    for optional_col in [
        '预测理论晚间车辆数',
        '理论晚间车辆数',
        '晚间实际车辆数',
        '晚间快照可用车辆',
        '预测库存_vs_真实理论误差',
        '预测库存_vs_真实实际误差',
        '预测库存_vs_晚间快照误差',
        '库存综合误差',
        '总综合误差',
    ]:
        if optional_col in summary_df.columns:
            output_columns.append(optional_col)
    if intervention_col:
        output_columns.append(intervention_col)

    return summary_df[output_columns].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_date', default='2026-02-01')
    parser.add_argument('--report_file', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--version_tag', default=None)
    parser.add_argument('--extra_tag', default=None)
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()

    version_tag = resolve_version_tag(PROJECT_ROOT, args.version_tag, [args.report_file, SCRIPT_DIR])
    if args.report_file:
        report_file = resolve_project_path(PROJECT_ROOT, args.report_file)
    else:
        report_file = resolve_default_report_file(version_tag, args.target_date, args.extra_tag)

    if args.output_dir:
        output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    else:
        output_dir = os.path.dirname(report_file)
    ensure_analysis_dir(output_dir)

    report_df = pd.read_csv(report_file)
    summary_df = build_summary_dataframe(report_df)
    if summary_df.empty:
        raise ValueError(f'报告为空: {report_file}')

    station_col = '站点名称'
    best_df = summary_df.head(args.top_k).copy()
    worst_df = summary_df.tail(args.top_k).sort_values('总综合误差', ascending=False).copy()

    summary_csv = os.path.join(output_dir, '站点预测表现中文总结表.csv')
    best_csv = os.path.join(output_dir, f'预测较准站点TOP{args.top_k}.csv')
    worst_csv = os.path.join(output_dir, f'偏差最大站点TOP{args.top_k}.csv')
    summary_json = os.path.join(output_dir, '站点预测表现摘要.json')

    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    best_df.to_csv(best_csv, index=False, encoding='utf-8-sig')
    worst_df.to_csv(worst_csv, index=False, encoding='utf-8-sig')

    summary = {
        'report_file': report_file,
        'version_tag': version_tag,
        'top_k': args.top_k,
        '有clean标签站点数': int(len(summary_df)),
        '流量综合误差平均值': float(summary_df['流量综合误差'].mean()),
        '库存综合误差平均值': float(summary_df['库存综合误差'].mean()) if '库存综合误差' in summary_df.columns else None,
        '总综合误差平均值': float(summary_df['总综合误差'].mean()),
        '最好站点前三': best_df[[station_col, '总综合误差']].head(3).to_dict(orient='records'),
        '最差站点前三': worst_df[[station_col, '总综合误差']].head(3).to_dict(orient='records'),
    }
    with open(summary_json, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    build_and_save_analysis_registry(PROJECT_ROOT)
    print('Saved summary table to:', summary_csv)
    print('Saved best stations to:', best_csv)
    print('Saved worst stations to:', worst_csv)
    print('Saved summary json to:', summary_json)


if __name__ == '__main__':
    main()
