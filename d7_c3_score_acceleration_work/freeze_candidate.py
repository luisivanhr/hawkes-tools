"""Write-once candidate/evidence lock before complete frozen-input replay."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    target=HERE/'CANDIDATE_FREEZE.json'
    if target.exists():raise FileExistsError(target)
    benchmarks=['first_candidate_benchmark.json','representative_benchmark.json','worker_benchmark.json']
    for name in benchmarks:
        if read_json(HERE/'evidence'/name)['status']!='EXACT':raise RuntimeError('benchmark not exact: '+name)
    if read_json(HERE/'evidence'/'edge_fixtures.json')['status']!='EXACT':raise RuntimeError('edge fixtures not exact')
    evidence=benchmarks+['edge_fixtures.json','baseline_profile.json','cold_baseline_0.json','cold_baseline_1.json','cold_candidate_0.json','cold_candidate_1.json']
    sources={p.name:sha256(p) for p in HERE.glob('*.py')}
    import scipy
    write_json(target,dict(status='FROZEN_FOR_ONE_COMPLETE_REPLAY',frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
               mergeable_modules=['vector_score.py','exact_trace_density.py','directional_score_reference.py'],source_sha256=sources,
               contract_sha256=sha256(HERE/'EXACTNESS_CONTRACT.md'),prereplay_evidence_sha256={n:sha256(HERE/'evidence'/n) for n in evidence},
               reference_pins=PINS,runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,blas_threads=1),
               task_settings=dict(model='gpt-5.6-sol',reasoning_effort='ultra',verification='Parent verified actual turn metadata and sent confirmation.',
                                  saved_service_tier='priority',effective_task_speed='not exposed by creation API; not independently confirmed',global_settings_changed=False),
               no_sampler=True,no_new_scientific_experiment=True))
    print('FROZEN',sha256(target),flush=True)

if __name__=='__main__':main()
