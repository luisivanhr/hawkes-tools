"""Bounded saved-input process-pool comparison; no sampler/default changes."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

_S={}
def init(reference):
    start=time.perf_counter();common,adapter,baseline=setup(reference)
    module=load_module('worker_benchmark_candidate',HERE/'vector_score.py')
    inputs=load_role(common,'training')
    _S.update(reference=Path(reference),adapter=adapter,module=module,inputs=inputs,initialization_seconds=time.perf_counter()-start)

def job(name):
    import psutil
    tick=time.perf_counter();archive,raw=restore_archive(_S['reference']/'raw'/name,_S['adapter'])
    receipt=read_json(_S['reference']/'receipts'/Path(name).with_suffix('.json'))
    if sha256(_S['reference']/'raw'/name)!=receipt['raw_sha256']:raise RuntimeError('raw drift')
    if _S['adapter'].score_adapter.trace_archive_sha256(archive)!=receipt['diagnostics']['trace_archive_sha256']:raise RuntimeError('trace mismatch')
    start=time.perf_counter();arrays,diag=evaluate(_S['module'],raw,archive,_S['inputs']);seconds=time.perf_counter()-start
    check=arrays_check(arrays,raw);exact=check['exact'] and diag==receipt['diagnostics']
    if not exact:raise RuntimeError('worker replay differs')
    return dict(raw=name,exact=exact,evaluation_seconds=seconds,job_seconds=time.perf_counter()-tick,
                initialization_seconds=_S['initialization_seconds'],worker_pid=os.getpid(),memory=psutil.Process().memory_info()._asdict(),
                payload_digest=hashlib.sha256(b''.join(arrays[k].tobytes() for k in sorted(arrays))).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--rounds',type=int,default=2);p.add_argument('--output',default='evidence/worker_benchmark.json');args=p.parse_args()
    import psutil
    memory_before=psutil.virtual_memory()._asdict()
    if memory_before['available']<3*1024**3:raise RuntimeError('less than 3GiB available; no four-worker benchmark')
    names=[f'training_shared_S00_R{r}_C{c:02d}_A0.npz' for c in range(4) for r in range(4)]
    rows=[]
    configurations=[('fresh_2',2,True),('persistent_2',2,False),('persistent_4',4,False)]
    for repeat in range(args.rounds):
        for label,workers,fresh in configurations if repeat%2==0 else list(reversed(configurations)):
            start=time.perf_counter();results=[]
            batches=[names[:8],names[8:]] if fresh else [names]
            for batch in batches:
                with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(args.reference,),max_tasks_per_child=4 if fresh else None) as pool:
                    # map delivers results in original job order, regardless
                    # of independent worker completion order.
                    results.extend(pool.map(job,batch))
            elapsed=time.perf_counter()-start
            row=dict(configuration=label,repeat=repeat,wall_seconds=elapsed,results=results,stable_payload_digest=hashlib.sha256(''.join(x['payload_digest'] for x in results).encode()).hexdigest())
            rows.append(row);write_json(HERE/args.output,dict(status='IN_PROGRESS',rows=rows,memory_before=memory_before))
            print(json.dumps(dict(configuration=label,repeat=repeat,wall_seconds=elapsed,pids=len({x['worker_pid'] for x in results}))),flush=True)
    exact=len({x['stable_payload_digest'] for x in rows})==1
    write_json(HERE/args.output,dict(status='EXACT' if exact else 'MISMATCH',rows=rows,memory_before=memory_before,memory_after=psutil.virtual_memory()._asdict(),
               scope='16 existing archives, repeated identically. Single-thread BLAS. Fresh 2-worker pools per eight jobs with max_tasks_per_child=4 versus persistent 2/4. Production executor and seed mapping unchanged; no claim of fresh-sampler stability.'))

if __name__=='__main__':main()
