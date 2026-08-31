"""Measure a first evaluator call in a new interpreter, without profiling."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
PROCESS_START=time.perf_counter()
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--implementation',choices=('baseline','candidate'),required=True);p.add_argument('--output',required=True);args=p.parse_args()
    common,adapter,baseline=setup(args.reference);inputs=load_role(common,'training')
    module=baseline if args.implementation=='baseline' else load_module('cold_candidate',HERE/'vector_score.py')
    initialization=time.perf_counter()-PROCESS_START
    name='training_shared_S00_R0_C00_A0.npz';tick=time.perf_counter();archive,raw=restore_archive(Path(args.reference)/'raw'/name,adapter);io=time.perf_counter()-tick
    receipt=read_json(Path(args.reference)/'receipts'/Path(name).with_suffix('.json'))
    if sha256(Path(args.reference)/'raw'/name)!=receipt['raw_sha256']:raise RuntimeError('input mismatch')
    tick=time.perf_counter();arrays,diag=evaluate(module,raw,archive,inputs);elapsed=time.perf_counter()-tick
    check=arrays_check(arrays,raw);exact=check['exact'] and diag==receipt['diagnostics']
    import psutil
    out=dict(implementation=args.implementation,exact=exact,initialization_seconds=initialization,io_reconstruction_seconds=io,
             first_evaluation_seconds=elapsed,total_script_seconds=time.perf_counter()-PROCESS_START,memory=psutil.Process().memory_info()._asdict(),comparison=check,
             scope='Fresh Python interpreter; pre-existing filesystem/compiler caches retained. Not a forced OS disk-cache flush.')
    write_json(HERE/args.output,out);print(json.dumps({k:v for k,v in out.items() if k not in ('comparison','memory')}),flush=True)
    if not exact:raise SystemExit(1)

if __name__=='__main__':main()
