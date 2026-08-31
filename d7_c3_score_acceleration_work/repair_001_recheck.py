"""One focused repair: preserve warning-as-error short-circuit ordering."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def capture(fn, policy, error_mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if policy=='all':warnings.simplefilter('error')
        elif policy=='runtime':warnings.filterwarnings('error',category=RuntimeWarning)
        elif policy=='underflow':warnings.filterwarnings('error',message='.*underflow.*',category=RuntimeWarning)
        elif policy=='overflow':warnings.filterwarnings('error',message='.*overflow.*',category=RuntimeWarning)
        elif policy=='irrelevant_category':warnings.filterwarnings('error',category=UserWarning)
        elif policy=='irrelevant_message':warnings.filterwarnings('error',message='this message does not occur',category=RuntimeWarning)
        with np.errstate(all=error_mode):
            try:result={'kind':'return','value_bits':np.float64(fn()).tobytes().hex()}
            except Exception as exc:result={'kind':'raise','type':type(exc).__name__,'message':str(exc)}
        result['warnings']=[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);args=p.parse_args()
    common,adapter,baseline=setup(args.reference);inputs=load_role(common,'training')
    after=load_module('repaired_density',HERE/'exact_trace_density.py')
    before=load_module('original_frozen_density',HERE/'evidence/repair_001/original_exact_trace_density.py')
    flags={'context_aware_warnings':sys.flags.context_aware_warnings,'error_filters_present':any(r[0]=='error' for r in warnings.filters),
           'filters':[repr(r) for r in warnings.filters],'numpy_errors':np.geterr()}
    if flags['context_aware_warnings']!=0 or flags['error_filters_present']:raise RuntimeError('default dispatch proof runtime differs')
    archive,raw=restore_archive(Path(args.reference)/'raw/training_shared_S00_R0_C00_A0.npz',adapter)
    original_trace=archive.traces[0];tiny=np.nextafter(0.,1.);rows=[];before_mismatch=None
    for pi,point in enumerate((np.full(25,.5),inputs['g'])):
        law,kernel,_=adapter.d7.model_from_normalized(point,inputs['context'])
        for ti,gaps in enumerate(((tiny,1e308),(1.8,tiny,1e308),(1.8,),(tiny,np.nan))):
            trace=SimpleNamespace(**vars(original_trace));trace.age=1.3;trace.internal_gaps=gaps
            for policy in ('all','runtime','underflow','overflow','irrelevant_category','irrelevant_message'):
                for mode in ('warn','raise'):
                    expected=capture(lambda:adapter.d7.base.trace_log_density(trace,law,kernel),policy,mode)
                    actual=capture(lambda:after.trace_log_density(trace,law,kernel,adapter.d7.base),policy,mode)
                    rows.append(dict(point=pi,gaps_case=ti,policy=policy,error_mode=mode,exact=actual==expected,actual=actual,expected=expected))
            if pi==0 and ti==0:
                old=capture(lambda:before.trace_log_density(trace,law,kernel,adapter.d7.base),'all','warn')
                expected=capture(lambda:adapter.d7.base.trace_log_density(trace,law,kernel),'all','warn')
                before_mismatch=dict(original_candidate=old,reference=expected,reproduced=old!=expected)
    # Both dispatch checks execute the actual original routine when selected.
    base=adapter.d7.base;original=base.trace_log_density;calls=[]
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    base.trace_log_density=counted
    try:
        law,kernel,_=adapter.d7.model_from_normalized(raw['point'],inputs['context'])
        value=after.trace_log_density(original_trace,law,kernel,base)
        default_fast=len(calls)==0
        with warnings.catch_warnings():
            warnings.filterwarnings('error',category=UserWarning)
            error_value=after.trace_log_density(original_trace,law,kernel,base)
        error_reference=len(calls)==1
    finally:base.trace_log_density=original
    dispatch_exact=array_check(np.asarray([value,error_value]),np.asarray([original(original_trace,law,kernel)]*2))['exact']
    exact=all(r['exact'] for r in rows) and before_mismatch['reproduced'] and default_fast and error_reference and dispatch_exact
    result=dict(status='FOCUSED_REPAIR_EXACT' if exact else 'FAIL',cases=len(rows),rows=rows,original_failure=before_mismatch,default_runtime=flags,
                default_fast_path_selected=default_fast,error_filter_reference_path_selected=error_reference,dispatch_values_exact=dispatch_exact,
                before_sha256=sha256(HERE/'evidence/repair_001/original_exact_trace_density.py'),after_sha256=sha256(HERE/'exact_trace_density.py'),
                source_sha256=sha256(__file__),scope='Retains original full-replay arithmetic when no error filter exists. Conservatively delegates to original reference for any error filter; no second complete replay.')
    write_json(HERE/'evidence/repair_001/warning_policy_recheck.json',result)
    print(json.dumps(dict(status=result['status'],cases=len(rows),failed=sum(not r['exact'] for r in rows),default_fast=default_fast,error_reference=error_reference)),flush=True)
    if not exact:raise SystemExit(1)

if __name__=='__main__':main()
