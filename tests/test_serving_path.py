"""No module on the serving path turns data into code or objects (#68, step 4).

The serving path is every module of this repository that `python -m glm_vx.server` can import. The
test finds it by following the import statements from glm_vx.server, also those inside functions,
so the GPU modules are on it. None of them may:
- import pickle, dill, cloudpickle, marshal, shelve, joblib, jinja2, yaml, torch or transformers;
- call eval, exec, compile or __import__, or importlib.import_module with a name it computes;
- call numpy.load with allow_pickle=True.

SGLang's CVE-2026-3059 and CVE-2026-3060 (CERT VU#665416: pickle.loads on a ZMQ socket) and
CVE-2026-5760 (CERT VU#915947: a model's chat template rendered by an unsandboxed Jinja2) are this
class, and docs/SECURITY.md lists the rest. glm_vx/generate.py renders chat templates with Jinja2's
sandbox, and glm_vx/disaggregated.py is a CPU demo on pipes; neither is on the path.

numpy itself imports pickle (np.load of object arrays), so `sys.modules` after an import shows
nothing; this test reads the source instead.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'glm_vx.server'
FORBIDDEN = {'pickle', '_pickle', 'cPickle', 'cloudpickle', 'dill', 'marshal', 'shelve', 'joblib', 'jinja2',
             'yaml', 'torch', 'transformers'}
BUILTINS = {'eval', 'exec', 'compile', '__import__'}


def source(module):
    path = ROOT.joinpath(*module.split('.'))
    for candidate in (path.with_suffix('.py'), path / '__init__.py'):
        if candidate.exists():
            return candidate


def imports(module, path, tree):
    package = module if path.name == '__init__.py' else module.rpartition('.')[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split('.')[:len(package.split('.')) - node.level + 1]
                base = '.'.join(parts + ([node.module] if node.module else []))
            else:
                base = node.module
            yield node, base
            for alias in node.names:
                yield node, f'{base}.{alias.name}'  # `from gpu import tp` imports a module


def serving_path():
    """{module: (path, tree)} for this repository's modules, and {external top-level name: [where]}."""
    found, external, todo = {}, {}, [ENTRY]
    while todo:
        module = todo.pop()
        if module in found:
            continue
        path = source(module)
        if path is None:
            continue
        tree = ast.parse(path.read_text(), str(path))
        found[module] = (path, tree)
        for node, name in imports(module, path, tree):
            if source(name) or source(name.split('.')[0]):
                todo.append(name)
            else:
                external.setdefault(name.split('.')[0], []).append(f'{path.relative_to(ROOT)}:{node.lineno}')
    return found, external


def test_the_walk_reaches_the_engine_the_link_and_the_gpu_modules():
    found, _ = serving_path()
    for module in ('glm_vx.server', 'glm_vx.link', 'glm_vx.scheduler', 'glm_vx.prefix', 'glm_vx.checkpoint',
                   'glm_vx.gguf_reader', 'gpu.tp_serve', 'gpu.resident', 'kernels.backend'):
        assert module in found, sorted(found)
    assert 'glm_vx.generate' not in found and 'glm_vx.disaggregated' not in found


def test_no_serving_module_imports_a_deserializer_or_a_template_engine():
    _, external = serving_path()
    bad = {name: where for name, where in external.items() if name in FORBIDDEN}
    assert not bad, bad


def test_no_serving_module_evaluates_code_or_loads_pickled_arrays():
    found, _ = serving_path()
    bad = []
    for module, (path, tree) in found.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f, where = node.func, f'{path.relative_to(ROOT)}:{node.lineno}'
            if isinstance(f, ast.Name) and f.id in BUILTINS:
                bad.append((where, f.id))
            if isinstance(f, ast.Attribute) and f.attr == 'import_module' and not (
                    node.args and isinstance(node.args[0], ast.Constant)):
                bad.append((where, 'import_module of a computed name'))
            if isinstance(f, ast.Attribute) and f.attr == 'load':
                for k in node.keywords:
                    if k.arg == 'allow_pickle' and not (isinstance(k.value, ast.Constant) and k.value.value is False):
                        bad.append((where, 'load(allow_pickle=...)'))
    assert not bad, bad
