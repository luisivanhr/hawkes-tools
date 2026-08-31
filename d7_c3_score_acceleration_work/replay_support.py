"""Read-only loading of frozen inputs. No latent generation entrypoint is used."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.cache' / 'numba')
os.environ['MPLCONFIGDIR'] = str(HERE / '.cache' / 'matplotlib')
import numpy as np

PINS = {
    'COMPLETE_ARTIFACT_MANIFEST.json': '3f7149b78f8fda72902642707e6f26b68690f4a096342e9d85b39066f7730525',
    'D7_C3_ORTHOGONALIZED_OPT_1_RESULTS.json': '7a00aaf46f8db86af6e1cac90d6a8cee68daa38748be1f56f0aa30fc51c8cfd5',
    'ORTHOGONALIZED_OPT_CONTRACT.json': 'eb8a41471de9153c0750240fac7ca0614c0a9b0c9a96488651536898b5e8a66b',
    'vector_score.py': '3d5299e2416184ff611be1a33ef665248f3f814cf7dedaf89fc6b88485bad190',
    'gradient_sampling.py': '707c9e611b05d1a318f2403c09041aeb0ace8244cc4813f0bcf25f264e658d90',
    'model_interface.py': '5426a0cc7b94613cdfedc10ba336280299d16e645436404f9ce766314ddbf349',
    'run_experiment.py': 'e2338108618b202e85807d1936a6da41f839ca4aec16098bcb85c8c4e2f9a1c8',
    'INDEPENDENT_SCIENTIFIC_AUDIT.md': 'a12416e77e15ed4cdab6811de28b475053e25140e1261b8fb495fa15726d7cd2',
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def ready(x):
    if isinstance(x, dict): return {str(k): ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [ready(v) for v in x]
    if isinstance(x, np.ndarray): return ready(x.tolist())
    if isinstance(x, np.generic): return ready(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def install_boundary():
    """Reject forbidden reads and any writes outside this isolated work directory."""
    root = os.path.normcase(str(HERE.resolve())) + os.sep
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            value = os.fsdecode(args[0]).lower().replace('\\', '/')
            base = value.rsplit('/', 1)[-1]
            if ('truth' in base or 'postlock_physical' in base or 'physical_diagnostic' in value
                    or 'generate_paths_once' in base or '/.codex/sessions/' in value):
                raise PermissionError('replay read boundary: ' + value)
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                p = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
                if not p.startswith(root): raise PermissionError('replay write boundary: ' + p)
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename'):
            for arg in args[:2] if event == 'os.rename' else args[:1]:
                if isinstance(arg, (str, bytes, os.PathLike)):
                    p = os.path.normcase(os.path.abspath(os.fsdecode(arg)))
                    if not (p + os.sep).startswith(root): raise PermissionError('replay mutation boundary: ' + p)
    sys.addaudithook(guard)

def setup(reference):
    reference = Path(reference).resolve()
    install_boundary()
    for rel, expected in PINS.items():
        if sha256(reference / rel) != expected: raise RuntimeError('frozen input drift: ' + rel)
    sys.path.insert(0, str(reference))
    common = load_module('orthogonal_common', reference / 'orthogonal_common.py')
    common.load_contract()  # Also verifies all twelve immutable source pins.
    adapter = common.safe_adapter()
    # build_chart_context uses WORK_ROOT only for mkdir of .numba_cache. Redirect
    # that non-numerical side effect; all immutable source roots stay untouched.
    adapter.d7.base.WORK_ROOT = HERE / '.cache'
    adapter.d7.base.WORK_ROOT.mkdir(exist_ok=True)
    def forbidden_sampler(*args, **kwargs):
        raise RuntimeError('latent/path generation is forbidden during saved-input replay')
    for module, names in ((adapter.score_adapter, ('build_centered_archive',)),
                          (adapter.d7, ('build_trace_archive',)),
                          (adapter.d7.base, ('build_trace_archive', 'sample_finite_root_trace'))):
        for name in names: setattr(module, name, forbidden_sampler)
    baseline = load_module('vector_score_baseline', reference / 'vector_score.py')
    return common, adapter, baseline

def load_role(common, role):
    a, context, g, paths, windows, indices, records = common.load_paths(role, selection_unlocked=role == 'selection')
    return dict(adapter=a, context=context, windows=windows, indices=indices, records=records, g=g)

def restore_archive(path, adapter):
    with np.load(path, allow_pickle=False) as z:
        raw = {key: np.array(z[key]) for key in z.files}
    n = raw['trace_age'].size
    traces = []
    for i in range(n):
        kw = {}
        for key in ('root_times', 'internal_gaps', 'cluster_event_times'):
            offsets = raw['trace_' + key + '_offsets']
            kw[key] = tuple(float(v) for v in raw['trace_' + key][offsets[i]:offsets[i+1]])
        crossing = raw['trace_crossing_distance'][i]
        kw.update(cutoff=float(raw['trace_cutoff'][i]), age=float(raw['trace_age'][i]),
                  crossing_distance=None if np.isnan(crossing) else float(crossing),
                  child_component_counts=tuple(int(v) for v in raw['trace_child_component_counts'][i]),
                  child_component_delay_sums=tuple(float(v) for v in raw['trace_child_component_delay_sums'][i]))
        traces.append(adapter.d7.base.FiniteRootTrace(**kw))
    archive = adapter.d7.TraceArchive(traces=tuple(traces), proposal_log_density=raw['proposal_log_density'],
                                   component_indices=np.zeros(n, dtype=np.int16), cutoff=float(raw['trace_cutoff'][0]), seed=int(raw['seed'][0]))
    return archive, raw

def array_check(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    same = a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')
    row = dict(exact=same, shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
    if not same:
        row.update(expected_shape=list(b.shape), expected_dtype=str(b.dtype), expected_sha256=hashlib.sha256(b.tobytes(order='C')).hexdigest())
        if a.shape == b.shape and a.dtype == b.dtype:
            row['different_bytes'] = sum(x != y for x, y in zip(a.tobytes(), b.tobytes()))
            if a.dtype.kind == 'f': row['maximum_absolute_difference'] = float(np.nanmax(np.abs(a-b)))
    return row

def arrays_check(actual, expected):
    checks = {k: array_check(v, expected[k]) for k,v in actual.items()}
    return dict(exact=all(v['exact'] for v in checks.values()), arrays=checks)

def evaluate(module, raw, archive, inputs):
    return module.evaluate_archive(raw['point'], archive=archive, clipping_cap=1e8,
                                   **{k:inputs[k] for k in ('context', 'windows', 'adapter')})
