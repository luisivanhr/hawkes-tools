"""One complete replay of saved raw traces through the accelerated public API.

Worker concurrency affects independent nodes only. Each original 128-trace
archive and all replicate/chunk reductions retain their original ordering.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

_STATE={}

def initialize(reference, candidate, output):
    common,adapter,baseline=setup(reference)
    module=load_module('vector_score_accelerated',HERE/candidate)
    import gradient_sampling
    _STATE.update(reference=Path(reference),common=common,adapter=adapter,module=module,
                  sampling=gradient_sampling,roles={},output=Path(output),contract=read_json(Path(reference)/'ORTHOGONALIZED_OPT_CONTRACT.json'),
                  manifest={x['path']:x for x in read_json(Path(reference)/'COMPLETE_ARTIFACT_MANIFEST.json')['entries']})

def pinned(rel):
    path=_STATE['reference']/rel
    expected=_STATE['manifest'][rel]['sha256']
    if sha256(path)!=expected:raise RuntimeError('manifest drift: '+rel)
    return path

def work(node_id):
    import psutil
    s=_STATE;start=time.perf_counter();role='selection' if node_id.startswith('selection') else 'training'
    if role not in s['roles']:s['roles'][role]=load_role(s['common'],role)
    inputs=s['roles'][role];assignment=read_json(pinned('assignments/'+node_id+'.json'))
    phase=assignment['phase'];slot=assignment['slot'];output=s['output'];chunks=[[] for _ in range(4)];diagnostics=[];seeds=[];jobs=[];levels=[]
    previous=None
    for target in (256,512):
        # Evaluation order cannot affect results; accumulation below still uses
        # the exact r-major / c-minor ordering in GradientService.ensure.
        for r in range(4):
            for c in range(len(chunks[r]),target//128):
                name=f'{phase}_S{slot:02d}_R{r}_C{c:02d}_A0'
                tick=time.perf_counter();path=pinned('raw/'+name+'.npz');receipt=read_json(pinned('receipts/'+name+'.json'))
                claim_path=pinned('claims/'+name+'.json')
                if receipt['raw_sha256']!=sha256(path) or receipt['claim_sha256']!=sha256(claim_path):raise RuntimeError('input chain mismatch')
                archive,raw=restore_archive(path,s['adapter'])
                if array_check(raw['point'],np.asarray(assignment['point'],dtype=np.float64))['exact'] is not True:raise RuntimeError('target mismatch')
                trace_digest=s['adapter'].score_adapter.trace_archive_sha256(archive)
                if trace_digest!=receipt['diagnostics']['trace_archive_sha256'] or archive.seed!=receipt['seed']:raise RuntimeError('saved trace identity mismatch')
                io=time.perf_counter()-tick;tick=time.perf_counter()
                arrays,diag=evaluate(s['module'],raw,archive,inputs)
                elapsed=time.perf_counter()-tick;check=arrays_check(arrays,raw);diag_exact=diag==receipt['diagnostics']
                job=dict(raw=name+'.npz',raw_sha256=receipt['raw_sha256'],seed=archive.seed,trace_archive_sha256=trace_digest,
                         input_load_verify_seconds=io,evaluation_seconds=elapsed,comparison=check,diagnostics_exact=diag_exact,fresh_diagnostics=diag)
                jobs.append(job)
                if not check['exact'] or not diag_exact:
                    write_json(output/'failures'/f'{name}.json',job);raise RuntimeError('raw replay mismatch: '+name)
                chunks[r].append({k:arrays[k] for k in ('log_weights','complete_gradients','trace_gradients')})
                del arrays,raw,archive
        # Match original receipt/seed ordering, including its 256 prefix.
        diagnostics=[];seeds=[]
        for r in range(4):
            for c in range(target//128):
                receipt=read_json(s['reference']/'receipts'/f'{phase}_S{slot:02d}_R{r}_C{c:02d}_A0.json')
                job_name=f'{phase}_S{slot:02d}_R{r}_C{c:02d}_A0.npz'
                entry=next(x for x in jobs if x['raw']==job_name)
                diagnostics.append(entry['fresh_diagnostics'])
                seeds.append(entry['seed'])
        tick=time.perf_counter();summary,arrays=s['module'].summarize(chunks,inputs['indices']);reduction_time=time.perf_counter()-tick
        stem=f'{node_id}_LEVEL_{target:04d}';saved=read_json(pinned('nodes/'+stem+'.json'))
        with np.load(pinned('nodes/'+stem+'.npz'),allow_pickle=False) as z:expected={k:z[k] for k in z.files}
        if set(arrays)!=set(expected):raise RuntimeError('node array keys differ')
        check=arrays_check(arrays,expected)
        checks=s['sampling'].technical_checks(summary,diagnostics,s['contract']['sampling'])
        quality=bool(target>=s['contract']['sampling']['minimum_terminal_traces_per_replicate'] and previous is not None and all(checks.values()))
        summary.update(node_id=node_id,candidate_id=assignment['candidate_id'],level=target,technical_checks=checks,quality_pass=quality,
                       seed_count=len(seeds),seeds=seeds,previous_gradient=None if previous is None else previous['gradient'],
                       previous_covariance=None if previous is None else previous['gradient_covariance'])
        scalar_checks={k:(ready(v)==saved[k] if isinstance(v,(dict,list)) or v is None else array_check(v,np.asarray(saved[k],dtype=np.asarray(v).dtype))['exact']) for k,v in summary.items()}
        if not check['exact'] or not all(scalar_checks.values()):
            write_json(output/'failures'/f'{stem}.json',dict(comparison=check,summary_checks=scalar_checks));raise RuntimeError('node replay mismatch: '+stem)
        (output/'nodes').mkdir(exist_ok=True,parents=True)
        np.savez_compressed(output/'nodes'/f'{stem}.npz',**arrays)
        write_json(output/'nodes'/f'{stem}.json',summary)
        levels.append(dict(level=target,comparison=check,summary_checks=scalar_checks,reduction_seconds=reduction_time))
        previous=arrays
    result=dict(node_id=node_id,exact=True,jobs=jobs,levels=levels,wall_seconds=time.perf_counter()-start,worker_pid=os.getpid(),memory=psutil.Process().memory_info()._asdict())
    write_json(output/'node_evidence'/f'{node_id}.json',result)
    return dict(node_id=node_id,exact=True,archives=len(jobs),wall_seconds=result['wall_seconds'],evaluation_seconds=sum(x['evaluation_seconds'] for x in jobs),memory=result['memory'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',default='vector_score.py')
    p.add_argument('--workers',type=int,choices=(1,2,4),default=4);p.add_argument('--output',default='evidence/full_replay')
    args=p.parse_args();reference=Path(args.reference).resolve();output=(HERE/args.output).resolve()
    if output.exists():raise FileExistsError('complete replay outputs are write-once: '+str(output))
    output.mkdir(parents=True)
    names=sorted(x.stem for x in (reference/'assignments').glob('*.json'))
    if len(names)!=39:raise RuntimeError('expected 39 frozen nodes')
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(reference),args.candidate,str(output))) as pool:
        futures={pool.submit(work,name):name for name in names}
        for future in as_completed(futures):
            try:row=future.result()
            except BaseException as exc:
                for pending in futures:pending.cancel()
                write_json(output/'FAILURE.json',dict(node=futures[future],exception=type(exc).__name__,message=str(exc),completed=rows));raise
            rows.append(row);print(json.dumps(dict(completed_nodes=len(rows),node=row['node_id'],seconds=row['wall_seconds'])),flush=True)
    write_json(output/'EXACTNESS_REPLAY.json',dict(status='RAW_AND_NODE_REPLAY_EXACT',node_count=len(rows),levels=2*len(rows),archive_count=sum(x['archives'] for x in rows),
               latent_traces=128*sum(x['archives'] for x in rows),new_latent_traces=0,workers=args.workers,wall_seconds=time.perf_counter()-start,
               candidate_sha256=sha256(HERE/args.candidate),rows=rows,source_pins=PINS))

if __name__=='__main__':main()
