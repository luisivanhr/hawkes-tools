"""Deterministic boundary/invalid-input parity, never stochastic fixtures."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def outcome(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            value=fn()
        return ('return',value)
    except Exception as exc:return ('raise',type(exc).__name__,str(exc))

def same_outcome(a,b):
    if a[0]!=b[0]:return False
    if a[0]=='raise':return a==b
    if isinstance(a[1],tuple):return all(array_check(x,y)['exact'] for x,y in zip(a[1],b[1],strict=True))
    return array_check(a[1],b[1])['exact']

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',default='evidence/edge_fixtures.json');args=p.parse_args()
    common,adapter,baseline=setup(args.reference);inputs=load_role(common,'training')
    candidate=load_module('vector_score_edge_candidate',HERE/'vector_score.py')
    from exact_trace_density import trace_log_density
    a=adapter.score_adapter.analytic;g=inputs['g'];context=inputs['context']
    traces=(a.FiniteRootTrace(cutoff=1536.,age=1.3,root_times=(-1.3,-3.1),internal_gaps=(1.8,),crossing_distance=1532.9,cluster_event_times=(-3.1,-2.6,-1.3,-.8),child_component_counts=(1,1),child_component_delay_sums=(.5,.5)),
            a.FiniteRootTrace(cutoff=1536.,age=2.1,root_times=(-2.1,-4.),internal_gaps=(1.9,),crossing_distance=1532.,cluster_event_times=(-4.,-3.6,-2.1,-1.8),child_component_counts=(1,1),child_component_delay_sums=(.4,.3)),
            a.FiniteRootTrace(cutoff=1536.,age=1536.,root_times=(),internal_gaps=(),crossing_distance=None,cluster_event_times=(),child_component_counts=(0,0),child_component_delay_sums=(0.,0.)))
    rows=[];points=[g.copy(),np.maximum(g,.02),np.full(25,.5)]
    valid_windows=[np.asarray([]),np.asarray([.4]),np.asarray([.4,1.2,2.7]),np.asarray([np.nextafter(0.,1.),np.nextafter(4.,0.)])]
    invalid_windows=[np.asarray([0.]),np.asarray([4.]),np.asarray([.4,.4]),np.asarray([1.,.5]),np.asarray([[-1.]]),np.asarray([np.nan]),np.asarray([np.inf])]
    smallest_positive=np.nextafter(0.,1.)
    for pi,point in enumerate(points):
        law,kernel,_=adapter.d7.model_from_normalized(point,context);evaluator=a.RenewalScoreEvaluator(law)
        proposal=np.asarray([adapter.d7.base.trace_log_density(t,law,kernel) for t in traces])
        archive=adapter.d7.TraceArchive(traces,proposal,np.zeros(len(traces),dtype=np.int16),1536.,0)
        windows=[dict(events=w,horizon=4.,path_index=i,window_index=0) for i,w in enumerate(valid_windows[:3])]
        aa=outcome(lambda:candidate.evaluate_archive(point,context=context,archive=archive,windows=windows,adapter=adapter,clipping_cap=1e8))
        bb=outcome(lambda:baseline.evaluate_archive(point,context=context,archive=archive,windows=windows,adapter=adapter,clipping_cap=1e8))
        exact=aa==bb if aa[0]=='raise' or bb[0]=='raise' else arrays_check(aa[1][0],bb[1][0])['exact'] and aa[1][1]==bb[1][1]
        rows.append(dict(kind='public_archive',point=pi,exact=exact,outcome=aa[0]))
        for wi,events in enumerate(valid_windows+invalid_windows):
            aa=outcome(lambda:candidate.conditional_batch(events,4.,traces,evaluator,kernel))
            bb=outcome(lambda:baseline.conditional_batch(events,4.,traces,evaluator,kernel))
            rows.append(dict(kind='window',point=pi,case=wi,exact=same_outcome(aa,bb),outcome=aa[0],exception=aa[1:] if aa[0]=='raise' else None))
        for mode in ('warn','ignore','raise'):
            with np.errstate(all=mode):
                for ti,gaps in enumerate(((1.8,),(),(-0.,),(0.,),(-1.,),(np.nan,),(np.inf,),
                                         (smallest_positive,1.8),(1e308,1.8),(1.8,np.nan))):
                    trace=SimpleNamespace(**vars(traces[0]));trace.internal_gaps=gaps
                    aa=outcome(lambda:trace_log_density(trace,law,kernel,adapter.d7.base))
                    bb=outcome(lambda:adapter.d7.base.trace_log_density(trace,law,kernel))
                    rows.append(dict(kind='gap_boundary',point=pi,mode=mode,case=ti,exact=same_outcome(aa,bb),outcome=aa[0],exception=aa[1:] if aa[0]=='raise' else None))
        for ti,trace in enumerate(traces):
            aa=outcome(lambda:trace_log_density(trace,law,kernel,adapter.d7.base));bb=outcome(lambda:adapter.d7.base.trace_log_density(trace,law,kernel))
            rows.append(dict(kind='complete_trace',point=pi,case=ti,exact=same_outcome(aa,bb)))
    payload=dict(status='EXACT' if all(x['exact'] for x in rows) else 'MISMATCH',cases=len(rows),rows=rows,
                 scope='Explicit deterministic traces (two inherited preflight fixtures plus empty-root fixture); saved G and deterministic legal points; no RNG calls. Seed 0 is only a fixture label, never a production seed.',
                 candidate_sha256=sha256(HERE/'vector_score.py'),helper_sha256=sha256(HERE/'exact_trace_density.py'))
    write_json(HERE/args.output,payload);print(json.dumps(dict(status=payload['status'],cases=len(rows),failed=[x for x in rows if not x['exact']]),default=str),flush=True)

if __name__=='__main__':main()
