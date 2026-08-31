"""Package the already certified local delivery without changing its evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    cert=read_json(HERE/'INDEPENDENT_CERTIFICATE.json')
    if cert['verdict']!='PASS' or cert['unresolved_findings']:raise RuntimeError('certification not complete')
    replay=read_json(HERE/'evidence/full_replay/EXACTNESS_REPLAY.json');history=read_json(HERE/'evidence/history_replay.json')
    bench=read_json(HERE/'evidence/repair_001/benchmark.json')
    medians={k:sum(float(np.median([t['seconds'] for t in r['trials'] if t['implementation']==k])) for r in bench['rows']) for k in ('baseline','candidate')}
    summary=dict(decision='CERTIFIED_EXACT_ON_FROZEN_REPLAY_WITH_FOCUSED_REPAIR',branch='codex/d7-c3-exact-score-acceleration',
                 runtime_modules=read_json(HERE/'FINAL_CANDIDATE_LOCK.json')['modules'],complete_replay_archives=624,latent_traces_replayed=79872,
                 node_levels=78,history_checks=history['comparisons'],scientific_decision=history['final_decision'],endpoint_ids=history['endpoint_ids'],
                 complete_replay_count=1,complete_replay_under_repaired_helper_hash=False,focused_repairs=1,
                 certification_basis=cert['scope'],repaired_evaluator_speedup=medians['baseline']/medians['candidate'],
                 original_representative_evaluator_speedup=3.653480940481463,production_end_to_end_speedup_measured=False,
                 full_replay_wall_seconds=replay['wall_seconds'],no_sampler=True,no_new_cohorts=True,no_merge_push_or_deletion=True,
                 artifacts={name:sha256(HERE/name) for name in ('SCORE_ACCELERATION_REPORT.md','INDEPENDENT_CERTIFICATE.md','INDEPENDENT_CERTIFICATE.json',
                            'CANDIDATE_FREEZE.json','FINAL_CANDIDATE_LOCK.json','REPAIR_001.json','evidence/full_replay/EXACTNESS_REPLAY.json',
                            'evidence/history_replay.json','evidence/replay_reconciliation.json')},
                 cleanup_requirement='Parent must preserve every retained file in DELIVERY_MANIFEST.json, including ignored node NPZ files/profiler binary, outside this worktree and verify integrated code before removing it. Keep original9e00/99d2/main dependencies; .cache is disposable.')
    write_json(HERE/'CERTIFICATION_SUMMARY.json',summary)
    entries=[]
    for path in sorted(HERE.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(HERE).as_posix()
        if '.cache' in path.relative_to(HERE).parts or '__pycache__' in path.relative_to(HERE).parts or rel=='DELIVERY_MANIFEST.json':continue
        ignored=(path.suffix=='.npz' and '/nodes/' in rel) or path.suffix=='.prof'
        entries.append(dict(path=rel,sha256=sha256(path),bytes=path.stat().st_size,committed_delivery=not ignored))
    write_json(HERE/'DELIVERY_MANIFEST.json',dict(scope='All retained files in isolated acceleration directory; disposable caches excluded; manifest self-hash intentionally omitted.',
               all_changed_paths=sorted([x['path'] for x in entries]+['DELIVERY_MANIFEST.json']),entries=entries,
               retained_file_count=len(entries)+1,ignored_evidence_count=sum(not x['committed_delivery'] for x in entries),
               total_retained_bytes_excluding_manifest=sum(x['bytes'] for x in entries),
               no_raw_archive_copies=True,excluded_disposable_roots=['.cache','__pycache__']))
    print(json.dumps(dict(decision=summary['decision'],repaired_speedup=summary['repaired_evaluator_speedup'],retained_files=len(entries)+1,
                          ignored_evidence=sum(not x['committed_delivery'] for x in entries),manifest_sha256=sha256(HERE/'DELIVERY_MANIFEST.json'))),flush=True)

if __name__=='__main__':main()
