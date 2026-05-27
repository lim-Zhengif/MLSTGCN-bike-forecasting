import json
import os
from typing import Iterable, Optional

import pandas as pd


ANALYSIS_DIRNAME = '分析结果'
VERSION_DIRNAME = '版本管理'
DEFAULT_MAINLINE_VERSION_TAG = '2026-04-07_优化前基线快照'
REGISTRY_XLSX_NAME = '单车模型结果汇总.xlsx'
REGISTRY_CSV_NAME = '单车模型结果汇总.csv'


def get_analysis_root(project_root: str) -> str:
    return os.path.join(project_root, ANALYSIS_DIRNAME)


def get_version_root(project_root: str) -> str:
    return os.path.join(project_root, VERSION_DIRNAME)


def list_version_tags(project_root: str):
    version_root = get_version_root(project_root)
    if not os.path.isdir(version_root):
        return []
    return sorted(
        entry
        for entry in os.listdir(version_root)
        if os.path.isdir(os.path.join(version_root, entry))
    )


def infer_version_tag_from_path(path: Optional[str], project_root: str) -> Optional[str]:
    if not path:
        return None

    abs_path = os.path.abspath(path)
    project_root = os.path.abspath(project_root)
    try:
        rel_path = os.path.relpath(abs_path, project_root)
    except ValueError:
        return None

    parts = rel_path.split(os.sep)
    version_tags = set(list_version_tags(project_root))
    if len(parts) >= 2 and parts[0] == VERSION_DIRNAME and parts[1] in version_tags:
        return parts[1]
    if len(parts) >= 2 and parts[0] == ANALYSIS_DIRNAME and parts[1] in version_tags:
        return parts[1]
    return None


def resolve_version_tag(
    project_root: str,
    explicit_version_tag: Optional[str] = None,
    candidate_paths: Optional[Iterable[str]] = None,
) -> str:
    if explicit_version_tag:
        return explicit_version_tag
    for path in candidate_paths or []:
        version_tag = infer_version_tag_from_path(path, project_root)
        if version_tag:
            return version_tag
    return DEFAULT_MAINLINE_VERSION_TAG


def build_analysis_task_dir(
    project_root: str,
    version_tag: str,
    task_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    extra_tag: Optional[str] = None,
    analysis_root: Optional[str] = None,
) -> str:
    folder_name = task_type
    if start_date and end_date:
        if start_date == end_date:
            folder_name = f'{task_type}_{start_date}'
        else:
            folder_name = f'{task_type}_{start_date}_到_{end_date}'
    elif start_date:
        folder_name = f'{task_type}_{start_date}'

    if extra_tag:
        folder_name = f'{folder_name}_{extra_tag}'

    analysis_root = analysis_root or get_analysis_root(project_root)
    return os.path.join(analysis_root, version_tag, folder_name)


def ensure_analysis_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_read_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception:
        return None


def _build_base_row(version_tag: str, task_folder: str, result_type: str, result_path: str):
    return {
        '版本': version_tag,
        '任务目录': task_folder,
        '结果类型': result_type,
        'date_start': None,
        'date_end': None,
        'overall_mae_mean': None,
        'overall_rmse_mean': None,
        'overall_mape_mean': None,
        '日间骑出量_mae_mean': None,
        '日间骑入量_mae_mean': None,
        'hourly_out_mae_mean': None,
        'hourly_in_mae_mean': None,
        'hourly_net_mae_mean': None,
        'pred_S_min_mae': None,
        'pred_S_max_mae': None,
        'pred_end_inventory_mae': None,
        '预测净流量误差_abs': None,
        '库存综合误差平均值': None,
        '总综合误差平均值': None,
        'train_project': None,
        'run_command': None,
        'entry_script': None,
        'best_val_mae_epoch': None,
        'test_mae': None,
        'test_loss': None,
        'log_dir': None,
        'checkpoint': None,
        '备注': f'使用版本：{version_tag}',
        '结果路径': result_path,
    }


def _parse_eval_summary(data, version_tag: str, task_folder: str, current_root: str):
    row = _build_base_row(version_tag, task_folder, '预测评估汇总', current_root)
    row.update(
        {
            'date_start': data.get('date_start'),
            'date_end': data.get('date_end'),
            'overall_mae_mean': data.get('overall_mae_mean'),
            'overall_rmse_mean': data.get('overall_rmse_mean'),
            'overall_mape_mean': data.get('overall_mape_mean'),
            '日间骑出量_mae_mean': data.get('日间骑出量_mae_mean'),
            '日间骑入量_mae_mean': data.get('日间骑入量_mae_mean'),
            'hourly_out_mae_mean': data.get('hourly_out_mae_mean'),
            'hourly_in_mae_mean': data.get('hourly_in_mae_mean'),
            'hourly_net_mae_mean': data.get('hourly_net_mae_mean'),
            'checkpoint': data.get('checkpoint'),
            '备注': f"使用版本：{data.get('version_tag') or version_tag}；任务：{data.get('task_name') or task_folder}",
        }
    )
    return row


def _parse_inventory_summary(data, version_tag: str, task_folder: str, current_root: str):
    row = _build_base_row(version_tag, task_folder, '单日库存推演', current_root)
    row.update(
        {
            'date_start': data.get('日期'),
            'date_end': data.get('日期'),
            'pred_end_inventory_mae': data.get('平均预测库存_vs_晚间快照误差_abs'),
            '预测净流量误差_abs': data.get('平均预测净流量误差_abs'),
            '备注': f'使用版本：{version_tag}；单日库存推演摘要',
        }
    )
    return row


def _parse_station_summary(data, version_tag: str, task_folder: str, current_root: str):
    row = _build_base_row(version_tag, task_folder, '站点表现总结', current_root)
    row.update(
        {
            '库存综合误差平均值': data.get('库存综合误差平均值'),
            '总综合误差平均值': data.get('总综合误差平均值'),
            '备注': f"使用版本：{version_tag}；TOP{data.get('top_k') or ''}站点总结".rstrip(),
        }
    )
    return row


def _parse_safe_inventory_summary(data, version_tag: str, task_folder: str, current_root: str):
    row = _build_base_row(version_tag, task_folder, '安全库存区间', current_root)
    row.update(
        {
            'pred_S_min_mae': data.get('pred_S_min_mae'),
            'pred_S_max_mae': data.get('pred_S_max_mae'),
            'pred_end_inventory_mae': data.get('pred_end_inventory_mae'),
            '备注': f'使用版本：{version_tag}；安全库存区间评估',
        }
    )
    return row


def _parse_training_summary(data, version_tag: str, task_folder: str, current_root: str):
    row = _build_base_row(version_tag, task_folder, '训练结果', current_root)
    row.update(
        {
            'train_project': data.get('project'),
            'run_command': data.get('entry_command') or data.get('resolved_train_command'),
            'entry_script': data.get('entry_script'),
            'best_val_mae_epoch': data.get('best_val_mae_epoch'),
            'test_mae': data.get('test_mae'),
            'test_loss': data.get('test_loss'),
            'log_dir': data.get('log_dir'),
            'checkpoint': data.get('best_checkpoint') or data.get('last_checkpoint'),
            '备注': f"使用版本：{version_tag}；训练项目：{data.get('project')}",
        }
    )
    return row


def collect_analysis_registry(project_root: str, analysis_root: Optional[str] = None):
    analysis_root = analysis_root or get_analysis_root(project_root)
    rows = []
    version_tags = list_version_tags(project_root)
    if not os.path.isdir(analysis_root):
        return rows

    for version_tag in version_tags:
        version_dir = os.path.join(analysis_root, version_tag)
        if not os.path.isdir(version_dir):
            continue

        for current_root, _, files in os.walk(version_dir):
            if current_root == version_dir:
                continue

            task_folder = os.path.relpath(current_root, version_dir).split(os.sep)[0]
            if 'summary.json' in files:
                data = _safe_read_json(os.path.join(current_root, 'summary.json'))
                if data:
                    rows.append(_parse_eval_summary(data, version_tag, task_folder, current_root))

            if '摘要.json' in files:
                data = _safe_read_json(os.path.join(current_root, '摘要.json'))
                if data:
                    rows.append(_parse_inventory_summary(data, version_tag, task_folder, current_root))

            if '站点预测表现摘要.json' in files:
                data = _safe_read_json(os.path.join(current_root, '站点预测表现摘要.json'))
                if data:
                    rows.append(_parse_station_summary(data, version_tag, task_folder, current_root))

            if 'safe_inventory_interval_summary.json' in files:
                data = _safe_read_json(os.path.join(current_root, 'safe_inventory_interval_summary.json'))
                if data:
                    rows.append(_parse_safe_inventory_summary(data, version_tag, task_folder, current_root))

            if 'training_summary.json' in files:
                data = _safe_read_json(os.path.join(current_root, 'training_summary.json'))
                if data:
                    rows.append(_parse_training_summary(data, version_tag, task_folder, current_root))

    return rows


def build_and_save_analysis_registry(
    project_root: str,
    output_xlsx: Optional[str] = None,
    output_csv: Optional[str] = None,
):
    rows = collect_analysis_registry(project_root)
    registry_df = pd.DataFrame(rows)
    column_order = [
        '版本',
        '任务目录',
        '结果类型',
        'date_start',
        'date_end',
        'overall_mae_mean',
        'overall_rmse_mean',
        'overall_mape_mean',
        '日间骑出量_mae_mean',
        '日间骑入量_mae_mean',
        'hourly_out_mae_mean',
        'hourly_in_mae_mean',
        'hourly_net_mae_mean',
        'pred_S_min_mae',
        'pred_S_max_mae',
        'pred_end_inventory_mae',
        '预测净流量误差_abs',
        '库存综合误差平均值',
        '总综合误差平均值',
        'train_project',
        'run_command',
        'entry_script',
        'best_val_mae_epoch',
        'test_mae',
        'test_loss',
        'log_dir',
        'checkpoint',
        '备注',
        '结果路径',
    ]
    if registry_df.empty:
        registry_df = pd.DataFrame(columns=column_order)
    else:
        registry_df = registry_df[column_order]
        registry_df = registry_df.sort_values(
            by=['版本', '结果类型', 'date_start', '任务目录'],
            na_position='last',
        ).reset_index(drop=True)

    analysis_root = get_analysis_root(project_root)
    output_xlsx = output_xlsx or os.path.join(analysis_root, REGISTRY_XLSX_NAME)
    output_csv = output_csv or os.path.join(analysis_root, REGISTRY_CSV_NAME)
    os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)

    compare_df = registry_df.copy()
    if not compare_df.empty:
        compare_df['日期范围'] = compare_df.apply(
            lambda row: row['date_start']
            if row['date_start'] == row['date_end']
            else (
                f"{row['date_start']} 到 {row['date_end']}"
                if pd.notna(row['date_start']) and pd.notna(row['date_end'])
                else ''
            ),
            axis=1,
        )
        compare_df = compare_df[
            [
                '版本',
                '结果类型',
                '任务目录',
                '日期范围',
                'overall_mae_mean',
                'overall_rmse_mean',
                'overall_mape_mean',
                '日间骑出量_mae_mean',
                '日间骑入量_mae_mean',
                'hourly_out_mae_mean',
                'hourly_in_mae_mean',
                'hourly_net_mae_mean',
                'pred_S_min_mae',
                'pred_S_max_mae',
                'pred_end_inventory_mae',
                '预测净流量误差_abs',
                '库存综合误差平均值',
                '总综合误差平均值',
                'train_project',
                'run_command',
                'entry_script',
                'best_val_mae_epoch',
                'test_mae',
                'test_loss',
                'log_dir',
                '备注',
            ]
        ]
    else:
        compare_df = pd.DataFrame(
            columns=[
                '版本',
                '结果类型',
                '任务目录',
                '日期范围',
                'overall_mae_mean',
                'overall_rmse_mean',
                'overall_mape_mean',
                '日间骑出量_mae_mean',
                '日间骑入量_mae_mean',
                'hourly_out_mae_mean',
                'hourly_in_mae_mean',
                'hourly_net_mae_mean',
                'pred_S_min_mae',
                'pred_S_max_mae',
                'pred_end_inventory_mae',
                '预测净流量误差_abs',
                '库存综合误差平均值',
                '总综合误差平均值',
                'train_project',
                'run_command',
                'entry_script',
                'best_val_mae_epoch',
                'test_mae',
                'test_loss',
                'log_dir',
                '备注',
            ]
        )

    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        registry_df.to_excel(writer, index=False, sheet_name='结果总表')
        compare_df.to_excel(writer, index=False, sheet_name='版本对比')
    registry_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return registry_df, output_xlsx, output_csv
