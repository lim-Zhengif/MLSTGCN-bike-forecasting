import argparse
import glob
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


def detect_project_root(start_dir):
    current = os.path.abspath(start_dir)
    while True:
        if (
            os.path.isdir(os.path.join(current, 'data'))
            and os.path.isdir(os.path.join(current, 'models'))
            and os.path.isdir(os.path.join(current, 'datasets'))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(start_dir)
        current = parent


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)


def resolve_project_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def normalize_date(text):
    return pd.to_datetime(text).strftime('%Y-%m-%d')


def resolve_snapshot_file(snapshot_dir, target_date):
    date_token = pd.to_datetime(target_date).strftime('%Y%m%d')
    pattern = os.path.join(snapshot_dir, '*%s*.csv' % date_token)
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError('No snapshot file matched %s under %s' % (date_token, snapshot_dir))
    return matches[0]


def resolve_column(df, candidates, label):
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError('Missing %s. Columns: %s' % (label, list(df.columns)))


def load_snapshot(snapshot_file, mapping_stations, prefix):
    df = pd.read_csv(snapshot_file)
    station_col = resolve_column(df, ['站点名称', 'station_name', 'Station'], 'snapshot station column')
    bike_col = resolve_column(df, ['当前可用车辆', 'available_bikes'], 'snapshot bike count column')
    dock_col = resolve_column(df, ['当前可用空位', 'available_docks'], 'snapshot dock count column')
    cap_col = resolve_column(df, ['总桩数', 'capacity'], 'snapshot capacity column')
    status_col = resolve_column(df, ['运营状态', 'status'], 'snapshot status column')

    df[station_col] = df[station_col].astype(str)
    df = df[df[station_col].isin(mapping_stations)].copy()
    df = df.drop_duplicates(subset=[station_col], keep='last')
    df = df[[station_col, bike_col, dock_col, cap_col, status_col]]
    return df.rename(
        columns={
            station_col: '站点名称',
            bike_col: '%s快照可用车辆' % prefix,
            dock_col: '%s快照可用空位' % prefix,
            cap_col: '%s快照总桩数' % prefix,
            status_col: '%s快照运营状态' % prefix,
        }
    )


def clip_inventory(values, capacity):
    if pd.isna(capacity):
        return values
    return min(max(values, 0.0), float(capacity))


def resolve_default_prediction_file(version_tag, target_date, extra_tag):
    eval_dir = build_analysis_task_dir(
        project_root=PROJECT_ROOT,
        version_tag=version_tag,
        task_type='单日滚动预测',
        start_date=target_date,
        end_date=target_date,
        extra_tag=extra_tag,
    )
    return os.path.join(eval_dir, 'station_predictions.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_date', default='2026-02-01')
    parser.add_argument('--prediction_file', default=None)
    parser.add_argument('--mapping_file', default=os.path.join('data', 'graph', 'bike', 'selected_node_mapping.csv'))
    parser.add_argument('--clean_table', default=os.path.join('二月份数据处理', 'CLEAN_JC_BaseTable_2026-02-01_to_2026-02-28.csv'))
    parser.add_argument('--morning_dir', default=os.path.join('二月份数据处理', 'morning'))
    parser.add_argument('--evening_dir', default=os.path.join('二月份数据处理', 'evening'))
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--version_tag', default=None)
    parser.add_argument('--task_name', default='单日库存推演')
    parser.add_argument('--extra_tag', default=None)
    args = parser.parse_args()

    target_date = normalize_date(args.target_date)
    if args.prediction_file:
        args.prediction_file = resolve_project_path(PROJECT_ROOT, args.prediction_file)
    args.mapping_file = resolve_project_path(PROJECT_ROOT, args.mapping_file)
    args.clean_table = resolve_project_path(PROJECT_ROOT, args.clean_table)
    args.morning_dir = resolve_project_path(PROJECT_ROOT, args.morning_dir)
    args.evening_dir = resolve_project_path(PROJECT_ROOT, args.evening_dir)

    version_tag = resolve_version_tag(PROJECT_ROOT, args.version_tag, [args.prediction_file, SCRIPT_DIR])
    if args.prediction_file is None:
        args.prediction_file = resolve_default_prediction_file(version_tag, target_date, args.extra_tag)
    if args.output_dir:
        args.output_dir = resolve_project_path(PROJECT_ROOT, args.output_dir)
    else:
        args.output_dir = build_analysis_task_dir(
            project_root=PROJECT_ROOT,
            version_tag=version_tag,
            task_type=args.task_name,
            start_date=target_date,
            end_date=target_date,
            extra_tag=args.extra_tag,
        )
    ensure_analysis_dir(args.output_dir)

    mapping_df = pd.read_csv(args.mapping_file).sort_values('Node_ID').reset_index(drop=True)
    if 'Node_ID' not in mapping_df.columns:
        raise KeyError('Mapping file must contain Node_ID: %s' % args.mapping_file)
    station_col = resolve_column(mapping_df, ['站点名称', 'station_name', 'Station'], 'mapping station column')
    mapping_df = mapping_df.rename(columns={station_col: '站点名称'})
    mapping_df['站点名称'] = mapping_df['站点名称'].astype(str)
    mapping_stations = set(mapping_df['站点名称'])

    prediction_df = pd.read_csv(args.prediction_file)
    pred_date_col = resolve_column(prediction_df, ['日期', 'date', 'Date'], 'prediction date column')
    pred_station_col = resolve_column(prediction_df, ['站点名称', 'station_name', 'Station'], 'prediction station column')
    prediction_df = prediction_df.rename(columns={pred_date_col: '日期', pred_station_col: '站点名称'})
    prediction_df['日期'] = pd.to_datetime(prediction_df['日期'], errors='coerce').dt.strftime('%Y-%m-%d')
    prediction_df = prediction_df[prediction_df['日期'] == target_date].copy()
    if prediction_df.empty:
        raise ValueError('No predictions found for %s in %s' % (target_date, args.prediction_file))

    pred_out_col = resolve_column(prediction_df, ['pred_日间骑出量', 'pred_out'], 'pred out column')
    pred_in_col = resolve_column(prediction_df, ['pred_日间骑入量', 'pred_in'], 'pred in column')
    true_out_col = resolve_column(prediction_df, ['true_日间骑出量', 'true_out'], 'true out column')
    true_in_col = resolve_column(prediction_df, ['true_日间骑入量', 'true_in'], 'true in column')
    prediction_df['pred_日间净流量'] = prediction_df[pred_in_col] - prediction_df[pred_out_col]
    prediction_df['true_日间净流量'] = prediction_df[true_in_col] - prediction_df[true_out_col]

    clean_df = pd.read_csv(args.clean_table)
    clean_date_col = resolve_column(clean_df, ['日期', 'date', 'Date'], 'clean date column')
    clean_station_col = resolve_column(clean_df, ['站点名称', 'station_name', 'Station'], 'clean station column')
    clean_df = clean_df.rename(columns={clean_date_col: '日期', clean_station_col: '站点名称'})
    clean_df['日期'] = pd.to_datetime(clean_df['日期'], errors='coerce').dt.strftime('%Y-%m-%d')
    clean_df['站点名称'] = clean_df['站点名称'].astype(str)
    clean_df = clean_df[(clean_df['日期'] == target_date) & (clean_df['站点名称'].isin(mapping_stations))].copy()

    morning_file = resolve_snapshot_file(args.morning_dir, target_date)
    evening_file = resolve_snapshot_file(args.evening_dir, target_date)
    morning_df = load_snapshot(morning_file, mapping_stations, '早晨')
    evening_df = load_snapshot(evening_file, mapping_stations, '晚间')

    result_df = mapping_df[['Node_ID', '站点名称']].copy()
    result_df = result_df.merge(prediction_df.drop(columns=['日期']), on='站点名称', how='left')

    desired_clean_cols = [
        '站点名称', '早晨初始车辆数', '总桩数', '日间骑出量', '日间骑入量',
        '日间纯用户净流量', '理论晚间车辆数', '晚间实际车辆数', '官方白天干预量',
    ]
    available_clean_cols = [col for col in desired_clean_cols if col in clean_df.columns]
    result_df = result_df.merge(clean_df[available_clean_cols], on='站点名称', how='left')
    result_df = result_df.merge(morning_df, on='站点名称', how='left')
    result_df = result_df.merge(evening_df, on='站点名称', how='left')

    result_df['用于推演的早晨车辆数'] = result_df['早晨初始车辆数'].combine_first(result_df['早晨快照可用车辆'])
    result_df['早晨车辆数来源'] = np.where(
        result_df['早晨初始车辆数'].notna(),
        'clean表',
        np.where(result_df['早晨快照可用车辆'].notna(), 'morning快照', '缺失'),
    )
    result_df['用于推演的总桩数'] = (
        result_df['总桩数']
        .combine_first(result_df['早晨快照总桩数'])
        .combine_first(result_df['晚间快照总桩数'])
    )

    result_df['预测理论晚间车辆数_未裁剪'] = result_df['用于推演的早晨车辆数'] + result_df['pred_日间净流量']
    result_df['预测理论晚间车辆数'] = result_df.apply(
        lambda row: clip_inventory(row['预测理论晚间车辆数_未裁剪'], row['用于推演的总桩数'])
        if pd.notna(row['预测理论晚间车辆数_未裁剪']) else np.nan,
        axis=1,
    )

    result_df['预测库存_vs_真实理论误差'] = result_df['预测理论晚间车辆数'] - result_df['理论晚间车辆数']
    result_df['预测库存_vs_真实实际误差'] = result_df['预测理论晚间车辆数'] - result_df['晚间实际车辆数']
    result_df['预测库存_vs_晚间快照误差'] = result_df['预测理论晚间车辆数'] - result_df['晚间快照可用车辆']
    result_df['预测净流量误差'] = result_df['pred_日间净流量'] - result_df['日间纯用户净流量']
    result_df['是否有clean真实标签'] = result_df['理论晚间车辆数'].notna()
    result_df['是否有晚间快照'] = result_df['晚间快照可用车辆'].notna()

    report_csv = os.path.join(args.output_dir, '站点级单日库存推演对比表.csv')
    missing_csv = os.path.join(args.output_dir, '缺少clean真实标签的站点.csv')
    summary_json = os.path.join(args.output_dir, '摘要.json')
    notes_md = os.path.join(args.output_dir, '说明.md')

    result_df.to_csv(report_csv, index=False, encoding='utf-8-sig')
    result_df[~result_df['是否有clean真实标签']].to_csv(missing_csv, index=False, encoding='utf-8-sig')

    summary = {
        '日期': target_date,
        '站点总数': int(len(result_df)),
        '有clean真实标签站点数': int(result_df['是否有clean真实标签'].sum()),
        '有晚间快照站点数': int(result_df['是否有晚间快照'].sum()),
        '平均预测净流量误差_abs': float(result_df['预测净流量误差'].abs().dropna().mean()),
        '平均预测库存_vs_真实理论误差_abs': float(result_df['预测库存_vs_真实理论误差'].abs().dropna().mean()),
        '平均预测库存_vs_真实实际误差_abs': float(result_df['预测库存_vs_真实实际误差'].abs().dropna().mean()),
        '平均预测库存_vs_晚间快照误差_abs': float(result_df['预测库存_vs_晚间快照误差'].abs().dropna().mean()),
        'morning快照文件': morning_file,
        'evening快照文件': evening_file,
        'prediction_file': args.prediction_file,
        'clean_table': args.clean_table,
        'version_tag': version_tag,
    }
    with open(summary_json, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    notes = [
        '# 单日库存推演对比说明',
        '',
        '- `pred_*` 列来自模型预测结果。',
        '- `预测理论晚间车辆数 = 用于推演的早晨车辆数 + pred_日间骑入量 - pred_日间骑出量`。',
        '- 预测理论晚间车辆数已按 `[0, 总桩数]` 做裁剪，同时保留了未裁剪列。',
        '- 晚间快照更接近业务上真正关心的“当天最终站点状态”。',
    ]
    with open(notes_md, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(notes))

    build_and_save_analysis_registry(PROJECT_ROOT)
    print('Saved report to:', report_csv)
    print('Saved missing-station list to:', missing_csv)
    print('Saved summary to:', summary_json)
    print('Saved notes to:', notes_md)
    print('Summary:', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
