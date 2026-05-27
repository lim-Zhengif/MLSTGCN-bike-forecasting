import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from analysis_result_utils import build_and_save_analysis_registry, resolve_version_tag


def load_inputs(eval_dir):
    summary_path = os.path.join(eval_dir, 'summary.json')
    daily_path = os.path.join(eval_dir, 'daily_metrics.csv')
    station_path = os.path.join(eval_dir, 'station_predictions.csv')

    if not os.path.exists(summary_path):
        raise FileNotFoundError('Missing summary.json under %s' % eval_dir)
    if not os.path.exists(daily_path):
        raise FileNotFoundError('Missing daily_metrics.csv under %s' % eval_dir)
    if not os.path.exists(station_path):
        raise FileNotFoundError('Missing station_predictions.csv under %s' % eval_dir)

    with open(summary_path, 'r', encoding='utf-8') as fp:
        summary = json.load(fp)

    daily_df = pd.read_csv(daily_path)
    station_df = pd.read_csv(station_path)
    return summary, daily_df, station_df


def load_inventory_summary(path):
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError('Missing inventory summary json: %s' % path)
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def infer_target_names(station_df):
    target_names = []
    for column in station_df.columns:
        if column.startswith('abs_error_'):
            target_names.append(column.replace('abs_error_', '', 1))
    if not target_names:
        raise ValueError('No abs_error_* columns were found in station_predictions.csv')
    return target_names


def compute_station_metrics(station_df, target_names):
    error_cols = ['abs_error_%s' % target_name for target_name in target_names]
    station_df = station_df.copy()
    station_df['双目标平均绝对误差'] = station_df[error_cols].mean(axis=1)

    overall = {
        '站点_日期_双目标平均绝对误差均值': float(station_df['双目标平均绝对误差'].mean()),
        '站点_日期_双目标平均绝对误差中位数': float(station_df['双目标平均绝对误差'].median()),
        '站点_日期_双目标平均绝对误差<=5占比': float((station_df['双目标平均绝对误差'] <= 5).mean()),
        '站点_日期_双目标平均绝对误差<=10占比': float((station_df['双目标平均绝对误差'] <= 10).mean()),
    }

    per_station_df = (
        station_df.groupby('站点名称', as_index=False)['双目标平均绝对误差']
        .mean()
        .rename(columns={'双目标平均绝对误差': '10天站点平均误差'})
        .sort_values('10天站点平均误差')
        .reset_index(drop=True)
    )
    per_station_df['误差排名'] = np.arange(1, len(per_station_df) + 1)
    per_station_df = per_station_df[['误差排名', '站点名称', '10天站点平均误差']]

    best_station_df = per_station_df.head(10).copy()
    worst_station_df = per_station_df.tail(10).sort_values('10天站点平均误差', ascending=False).copy()
    return overall, per_station_df, best_station_df, worst_station_df


def classify_conclusion(summary, station_metrics, inventory_summary):
    overall_mae = summary['overall_mae_mean']
    within10 = station_metrics['站点_日期_双目标平均绝对误差<=10占比']
    inventory_error = None
    if inventory_summary is not None:
        inventory_error = inventory_summary.get('平均预测库存_vs_晚间快照误差_abs')

    if overall_mae <= 5.0 and within10 >= 0.85:
        level = '可直接作为较强的调度参考'
        detail = '整体误差已经较低，可用于站点层的日常调度建议，但仍建议保留人工复核。'
    elif overall_mae <= 8.0 and within10 >= 0.70:
        level = '可作为粗粒度调度参考'
        detail = '适合做次日热点站点预警、运力优先级排序和大方向调度，不建议直接作为唯一自动下发依据。'
    else:
        level = '暂不适合作为正式调度依据'
        detail = '更适合做趋势观察或辅助分析，单站级调度仍存在较大风险。'

    if inventory_error is not None:
        detail += ' 单日库存推演对晚间快照的平均误差约为 %.3f 辆，可作为库存推演可行性的辅助证据。' % inventory_error

    return level, detail


def build_conclusion_table(summary, daily_df, station_metrics, best_station_df, worst_station_df, inventory_summary):
    best_day = daily_df.sort_values('overall_mae').iloc[0]
    worst_day = daily_df.sort_values('overall_mae', ascending=False).iloc[0]
    conclusion_level, conclusion_detail = classify_conclusion(summary, station_metrics, inventory_summary)

    rows = [
        {
            '类别': '评估设置',
            '指标': '预测任务',
            '数值': 'H=1 滚动单步预测',
            '说明': '每天使用前 7 天历史，预测下一天的日间骑出量和日间骑入量。',
        },
        {
            '类别': '评估设置',
            '指标': '日期范围',
            '数值': '%s 到 %s' % (summary['date_start'], summary['date_end']),
            '说明': '共 %s 天，%s 个站点。' % (summary['num_days'], summary['num_stations']),
        },
        {
            '类别': '总体表现',
            '指标': 'overall_mae_mean',
            '数值': round(summary['overall_mae_mean'], 3),
            '说明': '双目标整体 MAE 均值，数值越低越好。',
        },
        {
            '类别': '总体表现',
            '指标': 'overall_rmse_mean',
            '数值': round(summary['overall_rmse_mean'], 3),
            '说明': '双目标整体 RMSE 均值，反映大误差惩罚。',
        },
        {
            '类别': '分目标表现',
            '指标': '日间骑出量_mae_mean',
            '数值': round(summary['日间骑出量_mae_mean'], 3),
            '说明': '平均来看，单站单天骑出量误差大约在这个量级。',
        },
        {
            '类别': '分目标表现',
            '指标': '日间骑入量_mae_mean',
            '数值': round(summary['日间骑入量_mae_mean'], 3),
            '说明': '平均来看，单站单天骑入量误差大约在这个量级。',
        },
        {
            '类别': '站点级稳定性',
            '指标': '站点-日期双目标平均误差<=5占比',
            '数值': '%.1f%%' % (station_metrics['站点_日期_双目标平均绝对误差<=5占比'] * 100),
            '说明': '约有这些站点-日期样本的双目标平均误差控制在 5 辆以内。',
        },
        {
            '类别': '站点级稳定性',
            '指标': '站点-日期双目标平均误差<=10占比',
            '数值': '%.1f%%' % (station_metrics['站点_日期_双目标平均绝对误差<=10占比'] * 100),
            '说明': '约有这些站点-日期样本的双目标平均误差控制在 10 辆以内。',
        },
        {
            '类别': '日期表现',
            '指标': '最好日期',
            '数值': '%s (overall_mae=%.3f)' % (best_day['日期'], best_day['overall_mae']),
            '说明': '10 天中表现最好的一天。',
        },
        {
            '类别': '日期表现',
            '指标': '最差日期',
            '数值': '%s (overall_mae=%.3f)' % (worst_day['日期'], worst_day['overall_mae']),
            '说明': '10 天中表现最难的一天。',
        },
        {
            '类别': '站点表现',
            '指标': '最稳定站点前三',
            '数值': ' / '.join(best_station_df['站点名称'].head(3).tolist()),
            '说明': '按 10 天平均误差从低到高排序。',
        },
        {
            '类别': '站点表现',
            '指标': '高风险站点前三',
            '数值': ' / '.join(worst_station_df['站点名称'].head(3).tolist()),
            '说明': '这些站点在 10 天里平均误差最高，需要单独关注。',
        },
    ]

    if inventory_summary is not None:
        rows.extend([
            {
                '类别': '库存推演',
                '指标': '2026-02-01 预测库存_vs_晚间快照误差',
                '数值': round(inventory_summary['平均预测库存_vs_晚间快照误差_abs'], 3),
                '说明': '把预测流量推成库存后，与真实晚间快照相比的平均误差。',
            },
            {
                '类别': '库存推演',
                '指标': '2026-02-01 预测净流量误差',
                '数值': round(inventory_summary['平均预测净流量误差_abs'], 3),
                '说明': '说明流量预测推库存的误差传导大致处于可解释范围。',
            },
        ])

    rows.extend([
        {
            '类别': '业务结论',
            '指标': '是否可用于调度参考',
            '数值': conclusion_level,
            '说明': conclusion_detail,
        },
        {
            '类别': '业务结论',
            '指标': '建议使用方式',
            '数值': '优先级排序 / 热点预警 / 人工辅助调度',
            '说明': '更适合辅助判断哪些站点第二天更可能缺车或淤车，而不是完全自动派单。',
        },
    ])

    return pd.DataFrame(rows), conclusion_level, conclusion_detail


def build_markdown(summary, daily_df, best_station_df, worst_station_df, station_metrics, inventory_summary, conclusion_level, conclusion_detail):
    best_days = daily_df.sort_values('overall_mae').head(3)
    worst_days = daily_df.sort_values('overall_mae', ascending=False).head(3)

    lines = [
        '# Bike 模型 10 天滚动评估中文结论',
        '',
        '## 核心结论',
        '',
        '- 结论等级：`%s`' % conclusion_level,
        '- 客观判断：%s' % conclusion_detail,
        '',
        '## 关键指标',
        '',
        '| 指标 | 数值 |',
        '| --- | ---: |',
        '| 日期范围 | %s 到 %s |' % (summary['date_start'], summary['date_end']),
        '| 站点数 | %s |' % summary['num_stations'],
        '| overall_mae_mean | %.3f |' % summary['overall_mae_mean'],
        '| overall_rmse_mean | %.3f |' % summary['overall_rmse_mean'],
        '| 日间骑出量_mae_mean | %.3f |' % summary['日间骑出量_mae_mean'],
        '| 日间骑入量_mae_mean | %.3f |' % summary['日间骑入量_mae_mean'],
        '| 站点-日期双目标平均误差<=5占比 | %.1f%% |' % (station_metrics['站点_日期_双目标平均绝对误差<=5占比'] * 100),
        '| 站点-日期双目标平均误差<=10占比 | %.1f%% |' % (station_metrics['站点_日期_双目标平均绝对误差<=10占比'] * 100),
    ]

    if inventory_summary is not None:
        lines.extend([
            '| 2026-02-01 预测库存_vs_晚间快照误差 | %.3f |' % inventory_summary['平均预测库存_vs_晚间快照误差_abs'],
            '| 2026-02-01 预测净流量误差 | %.3f |' % inventory_summary['平均预测净流量误差_abs'],
        ])

    lines.extend([
        '',
        '## 日期层观察',
        '',
        '### 表现最好前三天',
        '',
        '| 日期 | overall_mae | 日间骑出量_mae | 日间骑入量_mae |',
        '| --- | ---: | ---: | ---: |',
    ])

    for _, row in best_days.iterrows():
        lines.append('| %s | %.3f | %.3f | %.3f |' % (
            row['日期'],
            row['overall_mae'],
            row['日间骑出量_mae'],
            row['日间骑入量_mae'],
        ))

    lines.extend([
        '',
        '### 表现最难前三天',
        '',
        '| 日期 | overall_mae | 日间骑出量_mae | 日间骑入量_mae |',
        '| --- | ---: | ---: | ---: |',
    ])

    for _, row in worst_days.iterrows():
        lines.append('| %s | %.3f | %.3f | %.3f |' % (
            row['日期'],
            row['overall_mae'],
            row['日间骑出量_mae'],
            row['日间骑入量_mae'],
        ))

    lines.extend([
        '',
        '## 站点层观察',
        '',
        '### 10 天最稳定站点前三',
        '',
        '| 排名 | 站点名称 | 10天站点平均误差 |',
        '| --- | --- | ---: |',
    ])

    for _, row in best_station_df.head(3).iterrows():
        lines.append('| %s | %s | %.3f |' % (
            int(row['误差排名']),
            row['站点名称'],
            row['10天站点平均误差'],
        ))

    lines.extend([
        '',
        '### 10 天高风险站点前三',
        '',
        '| 排名 | 站点名称 | 10天站点平均误差 |',
        '| --- | --- | ---: |',
    ])

    for _, row in worst_station_df.head(3).iterrows():
        lines.append('| %s | %s | %.3f |' % (
            int(row['误差排名']),
            row['站点名称'],
            row['10天站点平均误差'],
        ))

    lines.extend([
        '',
        '## 使用建议',
        '',
        '- 可以用于：次日热点站点预警、补车优先级排序、人工调度前的快速筛查。',
        '- 暂不建议直接用于：完全自动化的单站级精细调度指令下发。',
        '- 主要原因：当前 10 天均值表现已经有参考价值，但站点间差异仍明显，高风险站点误差仍偏大。',
    ])

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', required=True, help='Directory containing summary.json, daily_metrics.csv, station_predictions.csv')
    parser.add_argument('--inventory_summary_json', default=None, help='Optional single-day inventory summary json for business context.')
    parser.add_argument('--output_dir', default=None, help='Directory to save Chinese summary files. Defaults to eval_dir.')
    args = parser.parse_args()

    args.eval_dir = os.path.abspath(args.eval_dir)
    if args.inventory_summary_json:
        args.inventory_summary_json = os.path.abspath(args.inventory_summary_json)
    output_dir = os.path.abspath(args.output_dir or args.eval_dir)
    os.makedirs(output_dir, exist_ok=True)
    version_tag = resolve_version_tag(PROJECT_ROOT, None, [args.eval_dir, args.inventory_summary_json, SCRIPT_DIR])

    summary, daily_df, station_df = load_inputs(args.eval_dir)
    inventory_summary = load_inventory_summary(args.inventory_summary_json)
    target_names = infer_target_names(station_df)
    station_metrics, per_station_df, best_station_df, worst_station_df = compute_station_metrics(station_df, target_names)

    conclusion_table_df, conclusion_level, conclusion_detail = build_conclusion_table(
        summary=summary,
        daily_df=daily_df,
        station_metrics=station_metrics,
        best_station_df=best_station_df,
        worst_station_df=worst_station_df,
        inventory_summary=inventory_summary,
    )
    markdown_text = build_markdown(
        summary=summary,
        daily_df=daily_df,
        best_station_df=best_station_df,
        worst_station_df=worst_station_df,
        station_metrics=station_metrics,
        inventory_summary=inventory_summary,
        conclusion_level=conclusion_level,
        conclusion_detail=conclusion_detail,
    )

    summary_json = {
        'eval_dir': args.eval_dir,
        'version_tag': version_tag,
        'conclusion_level': conclusion_level,
        'conclusion_detail': conclusion_detail,
        'station_metrics': station_metrics,
        'best_days_top3': daily_df.sort_values('overall_mae').head(3).to_dict(orient='records'),
        'worst_days_top3': daily_df.sort_values('overall_mae', ascending=False).head(3).to_dict(orient='records'),
        'best_stations_top10': best_station_df.to_dict(orient='records'),
        'worst_stations_top10': worst_station_df.to_dict(orient='records'),
        'inventory_summary': inventory_summary,
    }

    table_path = os.path.join(output_dir, '10天滚动评估中文结论表.csv')
    markdown_path = os.path.join(output_dir, '10天滚动评估中文结论.md')
    station_path = os.path.join(output_dir, '10天站点平均误差排序.csv')
    summary_path = os.path.join(output_dir, '10天滚动评估中文结论摘要.json')

    conclusion_table_df.to_csv(table_path, index=False, encoding='utf-8-sig')
    per_station_df.to_csv(station_path, index=False, encoding='utf-8-sig')
    with open(markdown_path, 'w', encoding='utf-8') as fp:
        fp.write(markdown_text)
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary_json, fp, ensure_ascii=False, indent=2)

    try:
        build_and_save_analysis_registry(PROJECT_ROOT)
    except ModuleNotFoundError as exc:
        if exc.name != 'openpyxl':
            raise
        print('Skip analysis registry Excel export because openpyxl is not installed.')
    print('Saved conclusion table to:', table_path)
    print('Saved markdown report to:', markdown_path)
    print('Saved station ranking to:', station_path)
    print('Saved summary json to:', summary_path)
    print('Conclusion:', conclusion_level)


if __name__ == '__main__':
    main()
