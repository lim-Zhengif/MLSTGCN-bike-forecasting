import os
import shutil

from analysis_result_utils import (
    DEFAULT_MAINLINE_VERSION_TAG,
    build_analysis_task_dir,
    build_and_save_analysis_registry,
    ensure_analysis_dir,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_ROOT = os.path.join(PROJECT_ROOT, '分析结果')
OPTIMIZED_VERSION_TAG = '2026-04-07_优化版_鲁棒损失_图稀疏化_星期嵌入'


LEGACY_MAPPINGS = [
    {
        'source': '2026-02-01到2026-02-10滚动预测',
        'version_tag': DEFAULT_MAINLINE_VERSION_TAG,
        'task_type': '滚动预测',
        'start_date': '2026-02-01',
        'end_date': '2026-02-10',
        'extra_tag': None,
    },
    {
        'source': '2026-02-01单日滚动预测',
        'version_tag': DEFAULT_MAINLINE_VERSION_TAG,
        'task_type': '单日滚动预测',
        'start_date': '2026-02-01',
        'end_date': '2026-02-01',
        'extra_tag': None,
    },
    {
        'source': '2026-02-01单日库存推演对比',
        'version_tag': DEFAULT_MAINLINE_VERSION_TAG,
        'task_type': '单日库存推演',
        'start_date': '2026-02-01',
        'end_date': '2026-02-01',
        'extra_tag': None,
    },
    {
        'source': '2026-04-08_2026-02-01单日滚动预测_优化版',
        'version_tag': OPTIMIZED_VERSION_TAG,
        'task_type': '单日滚动预测',
        'start_date': '2026-02-01',
        'end_date': '2026-02-01',
        'extra_tag': None,
    },
    {
        'source': '2026-04-08_2026-02-01单日库存推演_优化版',
        'version_tag': OPTIMIZED_VERSION_TAG,
        'task_type': '单日库存推演',
        'start_date': '2026-02-01',
        'end_date': '2026-02-01',
        'extra_tag': None,
    },
    {
        'source': '2026-04-08_新版优化模型_10天滚动预测',
        'version_tag': OPTIMIZED_VERSION_TAG,
        'task_type': '滚动预测',
        'start_date': '2026-02-01',
        'end_date': '2026-02-10',
        'extra_tag': '首次',
    },
    {
        'source': '2026-04-08_新版优化模型_10天滚动预测_修正log1p',
        'version_tag': OPTIMIZED_VERSION_TAG,
        'task_type': '滚动预测',
        'start_date': '2026-02-01',
        'end_date': '2026-02-10',
        'extra_tag': 'log1p修正',
    },
]


def move_directory_contents(source_dir: str, target_dir: str):
    ensure_analysis_dir(target_dir)
    for name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, name)
        target_path = os.path.join(target_dir, name)
        if os.path.exists(target_path):
            continue
        shutil.move(source_path, target_path)
    if os.path.isdir(source_dir) and not os.listdir(source_dir):
        os.rmdir(source_dir)


def main():
    if not os.path.isdir(ANALYSIS_ROOT):
        raise FileNotFoundError(f'分析结果目录不存在: {ANALYSIS_ROOT}')

    moved = []
    skipped = []
    for item in LEGACY_MAPPINGS:
        source_dir = os.path.join(ANALYSIS_ROOT, item['source'])
        if not os.path.isdir(source_dir):
            skipped.append(item['source'])
            continue

        target_dir = build_analysis_task_dir(
            project_root=PROJECT_ROOT,
            version_tag=item['version_tag'],
            task_type=item['task_type'],
            start_date=item['start_date'],
            end_date=item['end_date'],
            extra_tag=item['extra_tag'],
        )
        move_directory_contents(source_dir, target_dir)
        moved.append((item['source'], os.path.relpath(target_dir, ANALYSIS_ROOT)))

    registry_df, xlsx_path, csv_path = build_and_save_analysis_registry(PROJECT_ROOT)
    print('Moved legacy folders:')
    for source_name, target_rel in moved:
        print(f'  {source_name} -> {target_rel}')
    print('Skipped missing folders:')
    for source_name in skipped:
        print(f'  {source_name}')
    print('Registry rows:', len(registry_df))
    print('Registry xlsx:', xlsx_path)
    print('Registry csv:', csv_path)


if __name__ == '__main__':
    main()
