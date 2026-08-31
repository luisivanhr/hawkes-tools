"""Paired cold/warm saved-archive timings; exact bytes are mandatory."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--candidate',default='vector_score.py')
    p.add_argument('--raw',action='append')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--output',default='evidence/benchmark.json')
    args=p.parse_args()
    common,adapter,baseline=setup(args.reference)
    candidate=load_module('vector_score_candidate',HERE/args.candidate)
    names=args.raw or ['training_shared_S00_R0_C00_A0.npz','training_likelihood_only_S13_R3_C03_A0.npz',
                      'training_orthogonal_c3_S13_R2_C02_A0.npz','selection_shared_S00_R1_C01_A0.npz']
    role_inputs={};rows=[]
    import psutil
    for name in names:
        role='selection' if name.startswith('selection') else 'training'
        if role not in role_inputs: role_inputs[role]=load_role(common,role)
        inputs=role_inputs[role]
        start=time.perf_counter();archive,raw=restore_archive(Path(args.reference)/'raw'/name,adapter);io=time.perf_counter()-start
        receipt=read_json(Path(args.reference)/'receipts'/Path(name).with_suffix('.json'))
        if adapter.score_adapter.trace_archive_sha256(archive)!=receipt['diagnostics']['trace_archive_sha256']:raise RuntimeError('trace mismatch')
        row={'raw':name,'io_reconstruction_seconds':io,'raw_sha256':sha256(Path(args.reference)/'raw'/name),'trials':[]}
        for trial in range(args.rounds):
            for label,module in ([('baseline',baseline),('candidate',candidate)] if trial%2==0 else [('candidate',candidate),('baseline',baseline)]):
                start=time.perf_counter();arrays,diagnostics=evaluate(module,raw,archive,inputs);elapsed=time.perf_counter()-start
                check=arrays_check(arrays,raw);exact=check['exact'] and diagnostics==receipt['diagnostics']
                result=dict(implementation=label,trial=trial,seconds=elapsed,exact=exact,
                            memory=psutil.Process().memory_info()._asdict(),comparison=check,diagnostics_exact=diagnostics==receipt['diagnostics'])
                row['trials'].append(result)
                print(json.dumps(dict(raw=name,implementation=label,trial=trial,seconds=elapsed,exact=exact)),flush=True)
                if not exact:
                    rows.append(row);write_json(HERE/args.output,dict(status='MISMATCH',rows=rows));return
        rows.append(row);write_json(HERE/args.output,dict(status='IN_PROGRESS',rows=rows))
    write_json(HERE/args.output,dict(status='EXACT',candidate_sha256=sha256(HERE/args.candidate),rows=rows,
                                   timing_method='Same process, alternating order each repeat. First evaluation of each implementation/input is reported separately; no source-output lookup.',
                                   available_memory=psutil.virtual_memory().available))

if __name__=='__main__':main()
