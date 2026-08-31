"""Replay the frozen decision history from recomputed full node payloads.

This does not evaluate/synthesize traces or rerun unchanged G/C3 map solvers.
The raw-evaluator replay must separately certify the supplied node arrays and
summary weight diagnostics. Original optimizer functions consume these arrays;
saved G/C3 records are explicit, hash-pinned deterministic inputs.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_support as support
import numpy as np


EXCLUDED_FIELDS = frozenset((
    'runtime_seconds', 'elapsed_level_seconds', 'arrays_sha256', 'results_sha256',
))
SUMMARY_KEYS = (
    'sample_count_total', 'sample_count_per_replicate', 'gradient',
    'gradient_norm', 'maximum_coordinate_se', 'minimum_pooled_absolute_ess',
    'minimum_replicate_absolute_ess', 'maximum_pooled_single_weight_share',
    'maximum_replicate_single_weight_share',
    'maximum_pooled_top_point_one_percent_share', 'maximum_pooled_top_one_percent_share',
    'maximum_replicate_top_point_one_percent_share', 'maximum_replicate_top_one_percent_share',
    'all_finite',
)


class Checks:
    def __init__(self):
        self.comparisons = 0
        self.failures = []

    def require(self, condition, label, detail=None):
        self.comparisons += 1
        if not condition:
            self.failures.append({'path': label, 'detail': detail})

    def equal(self, actual, expected, label):
        """Strict decoded JSON equality, including float64 signed-zero bits."""
        a, b = support.ready(actual), support.ready(expected)
        self._equal(a, b, label)

    def _equal(self, a, b, path):
        self.comparisons += 1
        if isinstance(a, dict) and isinstance(b, dict):
            ka, kb = set(a) - EXCLUDED_FIELDS, set(b) - EXCLUDED_FIELDS
            if ka != kb:
                self.failures.append({'path': path, 'keys_actual': sorted(ka), 'keys_expected': sorted(kb)})
            for k in sorted(ka & kb):
                self._equal(a[k], b[k], path + '.' + k)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.failures.append({'path': path, 'length_actual': len(a), 'length_expected': len(b)})
            for i, (x, y) in enumerate(zip(a, b)):
                self._equal(x, y, f'{path}[{i}]')
        elif type(a) is not type(b) or (
            struct.pack('>d', a) != struct.pack('>d', b) if isinstance(a, float) and isinstance(b, float) else a != b
        ):
            self.failures.append({'path': path, 'actual': a, 'expected': b,
                                  'actual_type': type(a).__name__, 'expected_type': type(b).__name__})


class FrozenInputs:
    def __init__(self, root, checks):
        self.root, self.checks = Path(root), checks
        manifest = support.read_json(self.root / 'COMPLETE_ARTIFACT_MANIFEST.json')
        self.pins = {r['path'].replace('\\', '/'): r['sha256'] for r in manifest['entries']}
        self.used = {}

    def path(self, relative):
        key = str(relative).replace('\\', '/')
        path = self.root / key
        actual = support.sha256(path)
        if key not in self.pins or actual != self.pins[key]:
            raise RuntimeError('frozen history input drift: ' + key)
        self.used[key] = actual
        return path

    def json(self, relative):
        return support.read_json(self.path(relative))

    def arrays(self, relative):
        with np.load(self.path(relative), allow_pickle=False) as source:
            return {k: np.array(source[k]) for k in source.files}


class ReplayService:
    """The original bounded allocation/ensure predicates, without dispatch."""
    def __init__(self, frozen, node_dir, contract, lock, gradient_module, common, checks):
        self.frozen, self.node_dir = frozen, Path(node_dir)
        self.contract, self.lock, self.gradient_module = contract, lock, gradient_module
        self.common, self.checks = common, checks
        registry = frozen.json('LATENT_SEED_REGISTRY.json')
        self.rows = {(r['phase'], r['slot'], r['replicate'], r['chunk']): r for r in registry['rows']}
        self.nodes, self.snapshots = {}, {}
        self.next_slot = {p: 0 for p in contract['sampling']['phase_slots']}
        self.node_payload_hashes = {}
        self.levels_consumed = []
        self.ensure_returns = []

    def allocate(self, phase, point, purpose, directions=(), require_positive=False):
        slot = self.next_slot[phase]
        if slot >= self.contract['sampling']['phase_slots'][phase]:
            raise RuntimeError('replay allocation exceeds frozen budget: ' + phase)
        self.next_slot[phase] += 1
        node_id = f'{phase}_{slot:02d}'
        role = 'selection' if phase.startswith('selection') else 'training'
        assignment = dict(node_id=node_id, phase=phase, role=role, slot=slot,
                          point=np.asarray(point), candidate_id=self.common.candidate_id(point),
                          purpose=purpose, contract_sha256=self.lock['contract_sha256'])
        self.checks.equal(assignment, self.frozen.json(f'assignments/{node_id}.json'), 'assignment.' + node_id)
        node = dict(node_id=node_id, assignment=assignment, level=0, history=[], quality_pass=False)
        self.nodes[node_id] = node
        return self.ensure(node, directions, require_positive=require_positive)

    def ensure(self, node, directions=(), require_positive=False, force_level=0):
        assignment = node['assignment']
        if node['quality_pass'] and node['level'] >= force_level:
            existing = [self.gradient_module.direction_checks(node, np.asarray(d), self.contract) for d in directions]
            if all(x['quality_pass'] and (not require_positive or x['certified_positive']) for x in existing):
                self.ensure_returns.append(dict(node=node['node_id'], level=node['level'], reason='existing_qualified'))
                return node
        for level in self.contract['sampling']['progressive_traces_per_replicate']:
            if level <= node['level']:
                continue
            version = f"{node['node_id']}_LEVEL_{level:04d}"
            array_path, json_path = self.node_dir / (version + '.npz'), self.node_dir / (version + '.json')
            if not array_path.is_file() or not json_path.is_file():
                raise RuntimeError('replay requested an unavailable frozen node level: ' + version)
            with np.load(array_path, allow_pickle=False) as z:
                arrays = {k: np.array(z[k]) for k in z.files}
            expected_arrays = self.frozen.arrays('nodes/' + version + '.npz')
            self.checks.require(set(arrays) == set(expected_arrays), version + '.array_keys')
            for key in sorted(set(arrays) & set(expected_arrays)):
                result = support.array_check(arrays[key], expected_arrays[key])
                self.checks.require(result['exact'], version + '.' + key, result if not result['exact'] else None)
            supplied = support.read_json(json_path)
            summary = {key: supplied[key] for key in SUMMARY_KEYS}
            self.checks.equal(summary['gradient'], arrays['gradient'], version + '.summary_gradient')
            self.checks.require(bool(summary['all_finite']) == all(np.all(np.isfinite(a)) for a in arrays.values()), version + '.finite')
            diagnostics, seeds = [], []
            for replicate in range(4):
                for chunk in range(level // self.contract['sampling']['microchunk_size']):
                    row = self.rows[(assignment['phase'], assignment['slot'], replicate, chunk)]
                    completed = []
                    for attempt in row['attempts']:
                        if (self.frozen.root / attempt['receipt']).exists():
                            receipt = self.frozen.json(attempt['receipt'])
                            self.checks.equal(receipt['seed'], attempt['seed'], version + '.seed')
                            self.checks.equal(receipt['candidate_id'], assignment['candidate_id'], version + '.receipt_candidate')
                            completed.append((attempt, receipt))
                    if len(completed) != 1:
                        raise RuntimeError('replay requires exactly one completed attempt: ' + row['logical_id'])
                    attempt, receipt = completed[0]
                    diagnostics.append(receipt['diagnostics'])
                    seeds.append(attempt['seed'])
            technical = self.gradient_module.technical_checks(summary, diagnostics, self.contract['sampling'])
            previous = node.get('arrays')
            node.update(level=level, arrays=arrays,
                        previous_gradient=None if previous is None else previous['gradient'],
                        previous_covariance=None if previous is None else previous['gradient_covariance'])
            node['quality_pass'] = bool(level >= self.contract['sampling']['minimum_terminal_traces_per_replicate']
                                        and previous is not None and all(technical.values()))
            directional = [self.gradient_module.direction_checks(node, np.asarray(d), self.contract) for d in directions]
            eligible = node['quality_pass'] and all(x['quality_pass'] and (not require_positive or x['certified_positive']) for x in directional)
            summary.update(node_id=node['node_id'], candidate_id=assignment['candidate_id'], level=level,
                           technical_checks=technical, quality_pass=node['quality_pass'], directional_diagnostics=directional,
                           seed_count=len(seeds), seeds=seeds, previous_gradient=node['previous_gradient'],
                           previous_covariance=node['previous_covariance'])
            self.checks.equal(summary, self.frozen.json('nodes/' + version + '.json'), version + '.summary')
            node.update(summary=summary, arrays_path=str(Path('nodes') / array_path.name),
                        results_path=str(Path('nodes') / json_path.name),
                        arrays_sha256=support.sha256(array_path), results_sha256=support.sha256(json_path), version=version)
            self.snapshots[version] = dict(node)
            node['history'].append({key: node[key] for key in ('level', 'results_path', 'results_sha256', 'arrays_path', 'arrays_sha256')})
            self.node_payload_hashes[version] = dict(npz_sha256=node['arrays_sha256'], json_sha256=node['results_sha256'])
            self.levels_consumed.append(version)
            if eligible and level >= force_level:
                self.ensure_returns.append(dict(node=node['node_id'], level=level, reason='new_qualified'))
                break
        return node


class PinnedModel:
    """Unchanged nonlinear solver outputs are declared inputs, never recomputed."""
    def __init__(self, frozen, common, checks, role):
        self.frozen, self.common, self.checks, self.role = frozen, common, checks, role
        self.used_constraints = set()

    def full_constraints(self, point):
        key = self.common.candidate_id(point)
        chart = bool(np.all(point >= 0.) and np.all(point <= 1.))
        if not chart:
            return dict(candidate_id=key, chart_feasible=False, constraint_compatible=False, status='CHART_REJECTION'), {'point': point}
        prefix = f'constraints/{self.role}/{key}'
        row, arrays = self.frozen.json(prefix + '.json'), self.frozen.arrays(prefix + '.npz')
        self.checks.require(support.array_check(point, arrays['point'])['exact'], prefix + '.point')
        compatible = bool(row['empirical_G']['accepted'] and row['structural']['passed'] and all(row['numerical']['checks'].values()))
        self.checks.equal(compatible, row['constraint_compatible'], prefix + '.compatible_predicate')
        row['constraint_compatible'] = compatible
        self.used_constraints.add(prefix)
        return row, arrays

    def gradients(self, point, config, need_c3):
        key = self.common.candidate_id(point)
        matches = []
        for arm in ('likelihood_only', 'orthogonal_c3'):
            for update in (1, 2, 3):
                row = self.frozen.json(f'directions/{arm}_UPDATE_{update}.json')
                diag = row['gradient_diagnostics']
                if row['candidate_id'] == key and ('C3_descent' in diag) == need_c3:
                    matches.append(diag)
        if not matches:
            raise RuntimeError('missing frozen deterministic direction inputs: ' + key)
        result = copy.deepcopy(matches[0])
        for name in ('G_gradient', 'C3_gradient', 'C3_descent', 'C3_jacobian', 'C3_coarse_jacobian'):
            if name in result:
                result[name] = np.asarray(result[name], dtype=np.float64)
        return result


def verify_history(reference, node_dir, *, selfcheck=False):
    reference, node_dir = Path(reference).resolve(), Path(node_dir).resolve()
    if not selfcheck and node_dir == reference / 'nodes':
        raise ValueError('archived nodes are allowed only with --selfcheck; they are not evaluator replay')
    common, adapter, baseline = support.setup(reference)
    checks = Checks()
    frozen = FrozenInputs(reference, checks)
    # Bind the exact original directional routine for unchanged optimizer calls.
    sys.modules['vector_score'] = baseline
    optimizer = support.load_module('optimizer_core', frozen.path('optimizer_core.py'))
    gradient = support.load_module('gradient_sampling', frozen.path('gradient_sampling.py'))
    runner = support.load_module('run_experiment', frozen.path('run_experiment.py'))
    contract = frozen.json('ORTHOGONALIZED_OPT_CONTRACT.json')
    lock = frozen.json('ORTHOGONALIZED_OPT_CONTRACT_LOCK.json')
    expected_results = frozen.json('D7_C3_ORTHOGONALIZED_OPT_1_RESULTS.json')
    endpoints = frozen.arrays('LOCKED_ENDPOINT_ARRAYS.npz')
    g = endpoints['protected_G']
    service = ReplayService(frozen, node_dir, contract, lock, gradient, common, checks)
    training_model = PinnedModel(frozen, common, checks, 'training')
    selection_model = PinnedModel(frozen, common, checks, 'selection')
    written = []

    def capture(path, value, **kwargs):
        relative = Path(path).relative_to(reference).as_posix()
        checks.equal(value, frozen.json(relative), 'captured.' + relative)
        written.append(relative)

    def saved_selection_paths(role, *, selection_unlocked=False):
        if role != 'selection' or not selection_unlocked:
            raise PermissionError('history selection requires joint endpoint lock')
        receipt = frozen.json('SELECTION_DATA_ACCESS_RECEIPT.json')
        # No path/window value is consumed by the selection control flow; the
        # evaluator replay independently supplies the actual saved windows.
        return adapter, None, g, (), (), np.repeat(np.arange(16, dtype=np.int16), 4), receipt['records']

    original = {key: getattr(runner, key) for key in ('write_json', 'load_paths', 'FrozenModelInterface')}
    runner.write_json = capture
    runner.load_paths = saved_selection_paths
    runner.FrozenModelInterface = lambda adapter, context, g, paths, role: selection_model
    try:
        checks.require(training_model.full_constraints(g)[0]['constraint_compatible'], 'protected_G_compatible')
        construction = service.allocate('training_shared', g, 'shared_initial_full_complete_gradient')
        states = {arm: dict(point=g.copy(), accepted=[], trials=[], stop_reason=None, construction=construction,
                            training_gain=optimizer.combine_integrals([], lambda key: None)) for arm in runner.ARMS}
        shared_gradients = training_model.gradients(g, contract['direction'], True)
        prepared = {arm: runner.prepare_direction(arm, 1, states[arm], training_model, service, contract,
                                                  construction, shared_gradients) for arm in runner.ARMS}
        directions = [r['direction'] for r in prepared.values() if r['status'] == 'DIRECTION_LOCKED']
        base = service.allocate('training_shared', g, 'independent_shared_G_direction_validation',
                                directions=directions, require_positive=True) if directions else None
        for update in range(1, contract['maximum_accepted_updates_per_arm'] + 1):
            for arm in runner.ARMS:
                state = states[arm]
                if state['stop_reason'] is not None:
                    continue
                if update == 1:
                    prep, validation = prepared[arm], base
                else:
                    prep = runner.prepare_direction(arm, update, state, training_model, service, contract)
                    validation = service.allocate('training_' + arm, state['point'], f'independent_update_{update}_base_validation',
                                                  directions=[prep['direction']], require_positive=True) if prep['status'] == 'DIRECTION_LOCKED' else None
                if prep['status'] != 'DIRECTION_LOCKED':
                    state['stop_reason'] = prep['status']
                    continue
                construction_seeds = set(prep['construction']['summary']['seeds'])
                validation_seeds = set(validation['summary']['seeds'])
                checks.require(construction_seeds.isdisjoint(validation_seeds), f'{arm}.{update}.independent_validation_seeds')
                runner.training_update(arm, update, state, prep, validation, training_model, service, contract)
                checkpoint = dict(point=state['point'], candidate_id=common.candidate_id(state['point']),
                                  accepted_updates=len(state['accepted']), stop_reason=state['stop_reason'], training_gain=state['training_gain'])
                checks.equal(checkpoint, frozen.json(f'checkpoints/{arm}_AFTER_UPDATE_{update}.json'), f'checkpoint.{arm}.{update}')
        regenerated_endpoints = {}
        for arm in runner.ARMS:
            state = states[arm]
            if state['stop_reason'] is None:
                state['stop_reason'] = 'THREE_ACCEPTED_UPDATES_LIMIT'
            checks.require(support.array_check(state['point'], endpoints[arm])['exact'], 'endpoint_bytes.' + arm)
            regenerated_endpoints[arm] = dict(point=state['point'], candidate_id=common.candidate_id(state['point']),
                                             accepted_updates=len(state['accepted']), stop_reason=state['stop_reason'],
                                             training_gain=state['training_gain'], constraints=training_model.full_constraints(state['point'])[0])
        checks.equal(regenerated_endpoints, frozen.json('BOTH_TRAINING_ENDPOINTS_LOCK.json')['endpoints'], 'joint_endpoint_lock')
        # These original functions construct coefficient dicts in runtime node
        # order. Never sum the sorted coefficient dicts loaded from JSON.
        selection = runner.selection_comparison(states, g, service, contract)
        checks.equal(selection, expected_results['selection'], 'results.selection')
        accepted = sum(len(s['accepted']) for s in states.values())
        if accepted == 0:
            decision = 'NO_CERTIFIED_TRAINING_UPDATE'
        else:
            difference = selection['orthogonal_minus_likelihood_only']
            if difference['numerical_quality_pass'] and difference['numerical_interval'][0] > 0.:
                decision = 'ORTHOGONAL_ARM_SELECTION_ADVANTAGE'
            elif difference['numerical_quality_pass'] and difference['numerical_interval'][1] < 0.:
                decision = 'LIKELIHOOD_ONLY_ARM_SELECTION_ADVANTAGE'
            else:
                decision = 'PAIRED_DEVELOPMENT_COMPARISON_UNRESOLVED'
        checks.equal(decision, expected_results['decision'], 'final_decision')
        for arm in runner.ARMS:
            actual = {k: v for k, v in states[arm].items() if k != 'construction'}
            checks.equal(actual, expected_results['arms'][arm], 'results.arms.' + arm)
        checks.equal(service.next_slot, expected_results['node_slots_used'], 'allocation_counts')
        index = {key: runner.compact_node(node) for key, node in service.nodes.items()}
        checks.equal(index, frozen.json('NODE_INDEX.json'), 'node_index')
        checks.require(len(service.nodes) == 39 and len(service.levels_consumed) == 78, 'complete_39_nodes_78_levels')
        checks.require(all(service.levels_consumed.count(v) == 1 for v in service.levels_consumed), 'each_level_consumed_once')
        checks.require(all(n['level'] == 512 for n in service.nodes.values()), 'no_1024_expansion_predicate_triggered')
        trial_statuses = [t['status'] for s in states.values() for t in s['trials']]
        checks.require(trial_statuses.count('ACCEPTED') == 6 and trial_statuses.count('CHART_G_OR_C3_NUMERICS_REJECTED') == 1
                       and len(trial_statuses) == 7, 'six_acceptances_one_chart_rejection')
        left = selection['arms']['likelihood_only']['gain']
        right = selection['arms']['orthogonal_c3']['gain']
        shared = set(left['coefficient_vectors']) & set(right['coefficient_vectors'])
        checks.equal(sorted(shared), ['selection_shared_00_LEVEL_0512'], 'paired_common_G_identity')
    finally:
        for key, value in original.items():
            setattr(runner, key, value)
    return dict(
        status=('PASS_HARNESS_DEVELOPMENT_SELFCHECK' if selfcheck else 'PASS_HISTORY_FROM_RECOMPUTED_NODES') if not checks.failures else 'FAIL',
        exact=not checks.failures, selfcheck=selfcheck, reference=str(reference), nodes=str(node_dir),
        comparisons=checks.comparisons, failures=checks.failures, final_decision=decision,
        endpoint_ids={a: common.candidate_id(states[a]['point']) for a in runner.ARMS},
        node_count=len(service.nodes), level_count=len(service.levels_consumed),
        levels_consumed=service.levels_consumed, ensure_returns=service.ensure_returns,
        captured_original_outputs=written, supplied_node_hashes=service.node_payload_hashes,
        frozen_input_hashes=frozen.used, source_sha256=support.sha256(__file__),
        exact_scalar_rule='decoded JSON type/value equality; float64 IEEE bytes including signed zero; NPZ dtype/shape/C-order payload bytes',
        exclusions=sorted(EXCLUDED_FIELDS),
        dependency_boundary='G/C3 gradients, full-map arrays, structural and numerical diagnostics are immutable manifest-pinned inputs; their compatibility predicate is rerun, solvers are not. Supplied node arrays/weight diagnostics must separately be linked to raw evaluator replay. Receipt parity diagnostics are pinned input and must separately match evaluator replay.',
        entrypoints_called=['run_experiment.prepare_direction', 'run_experiment.training_update', 'run_experiment.selection_comparison',
                            'optimizer_core.construct_direction', 'optimizer_core.integrate_nodes', 'optimizer_core.combine_integrals',
                            'gradient_sampling.technical_checks', 'gradient_sampling.direction_checks'],
        production_entrypoint_called=False, sampler_called=False, new_paths_or_traces=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--nodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selfcheck', action='store_true', help='harness development only; archived nodes allowed')
    args = parser.parse_args()
    result = verify_history(args.reference, args.nodes, selfcheck=args.selfcheck)
    support.write_json(args.output, result)
    print(result['status'], 'comparisons=', result['comparisons'], 'failures=', len(result['failures']), flush=True)
    if not result['exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
