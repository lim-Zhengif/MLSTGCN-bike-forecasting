import importlib.util
import os


def _detect_project_root(start_dir):
    current = os.path.abspath(start_dir)
    while True:
        if (
            os.path.isdir(os.path.join(current, 'data'))
            and os.path.isdir(os.path.join(current, 'models'))
            and os.path.isdir(os.path.join(current, 'datasets'))
            and os.path.isfile(os.path.join(current, 'analysis_result_utils.py'))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise ModuleNotFoundError('Cannot locate project-root analysis_result_utils.py from %s' % start_dir)
        current = parent


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _detect_project_root(_SCRIPT_DIR)
_MODULE_PATH = os.path.join(_PROJECT_ROOT, 'analysis_result_utils.py')
_SPEC = importlib.util.spec_from_file_location('_project_analysis_result_utils', _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ModuleNotFoundError('Cannot load analysis_result_utils from %s' % _MODULE_PATH)

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


build_analysis_task_dir = _MODULE.build_analysis_task_dir
build_and_save_analysis_registry = _MODULE.build_and_save_analysis_registry
ensure_analysis_dir = _MODULE.ensure_analysis_dir
get_analysis_root = _MODULE.get_analysis_root
get_version_root = _MODULE.get_version_root
infer_version_tag_from_path = _MODULE.infer_version_tag_from_path
list_version_tags = _MODULE.list_version_tags
resolve_version_tag = _MODULE.resolve_version_tag
