"""Bind the single warning-policy repair to preserved complete replay evidence."""
from pathlib import Path
import difflib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    target=HERE/'REPAIR_001.json'
    if target.exists():raise FileExistsError(target)
    freeze=read_json(HERE/'CANDIDATE_FREEZE.json');before_path=HERE/'evidence/repair_001/original_exact_trace_density.py';after_path=HERE/'exact_trace_density.py'
    before=before_path.read_text();after=after_path.read_text()
    expected=before.replace('import math\n','import math\nimport warnings\n',1).replace(
        "    if not supported or any(mode not in ('ignore', 'warn') for mode in np.geterr().values()):\n",
        "    if (not supported or any(mode not in ('ignore', 'warn') for mode in np.geterr().values())\n            or any(rule[0] == 'error' for rule in warnings.filters)):\n",1)
    if after!=expected or sha256(before_path)!=freeze['source_sha256']['exact_trace_density.py']:raise RuntimeError('repair is not the exact bounded dispatch delta')
    evidence={'full_replay':'evidence/full_replay/EXACTNESS_REPLAY.json','history':'evidence/history_replay.json',
              'original_reconciliation':'evidence/replay_reconciliation.json','preservation':'evidence/source_preservation_after_replay.json',
              'warning_policy':'evidence/repair_001/warning_policy_recheck.json','edge_fixtures':'evidence/repair_001/edge_fixtures.json','benchmark':'evidence/repair_001/benchmark.json'}
    if read_json(HERE/evidence['warning_policy'])['status']!='FOCUSED_REPAIR_EXACT' or read_json(HERE/evidence['edge_fixtures'])['status']!='EXACT' or read_json(HERE/evidence['benchmark'])['status']!='EXACT':raise RuntimeError('focused checks failed')
    diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original_frozen/exact_trace_density.py',tofile='repaired/exact_trace_density.py'))
    (HERE/'evidence/repair_001/source.diff').write_text(diff,encoding='utf-8')
    payload=dict(status='FOCUSED_REPAIR_RECHECK_PASSED',repair_number=1,
                 reason='Warning-as-error policy could expose later vector-division overflow before earlier scalar underflow. Any error filter now selects the original reference before batching.',
                 original_candidate_freeze_sha256=sha256(HERE/'CANDIDATE_FREEZE.json'),changed_runtime_source='exact_trace_density.py',
                 before_path=str(before_path.relative_to(HERE)),before_sha256=sha256(before_path),after_sha256=sha256(after_path),
                 exact_source_delta_verified=True,source_diff_sha256=sha256(HERE/'evidence/repair_001/source.diff'),
                 evidence={k:dict(path=v,sha256=sha256(HERE/v)) for k,v in evidence.items()},
                 default_path_argument='Under the observed context_aware_warnings=0 policy with no error filters, the added condition is false; every original floating-point operation remains unchanged. With any error filter, evaluation delegates to the original reference before any batched gap work.',
                 complete_replay_count=1,complete_replay_under_repaired_hash=False,
                 focused_recheck='96 warning-policy/error-mode cases, 135 prior deterministic cases, two complete representative archives in two paired timing runs each; all exact.',
                 scope='Final candidate certification uses original full corpus replay plus the unchanged-default-path argument and focused repaired-branch tests. It does not claim a second 624-archive replay under the repaired hash.')
    write_json(target,payload)
    write_json(HERE/'FINAL_CANDIDATE_LOCK.json',dict(status='FROZEN_AFTER_ONE_FOCUSED_REPAIR',repair_receipt_sha256=sha256(target),
               original_freeze_sha256=sha256(HERE/'CANDIDATE_FREEZE.json'),modules={n:sha256(HERE/n) for n in freeze['mergeable_modules']},
               focused_test_source_sha256=sha256(HERE/'repair_001_recheck.py')))
    print('REPAIR SEALED',sha256(target),'FINAL LOCK',sha256(HERE/'FINAL_CANDIDATE_LOCK.json'),flush=True)

if __name__=='__main__':main()
