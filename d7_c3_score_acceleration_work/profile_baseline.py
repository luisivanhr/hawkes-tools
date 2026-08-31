"""Reproduce and profile one existing archive, without a sampler call."""
from __future__ import annotations
import argparse
import cProfile
from pathlib import Path
import pstats
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--raw', default='training_shared_S00_R0_C00_A0.npz')
    p.add_argument('--output', default='evidence/baseline_profile.json')
    args=p.parse_args()
    start=time.perf_counter()
    common,adapter,baseline=setup(args.reference)
    inputs=load_role(common, 'selection' if args.raw.startswith('selection') else 'training')
    init=time.perf_counter()-start
    raw_path=Path(args.reference)/'raw'/args.raw
    receipt=read_json(Path(args.reference)/'receipts'/Path(args.raw).with_suffix('.json'))
    start=time.perf_counter();archive,raw=restore_archive(raw_path,adapter);read_time=time.perf_counter()-start
    if adapter.score_adapter.trace_archive_sha256(archive)!=receipt['diagnostics']['trace_archive_sha256']:
        raise RuntimeError('trace reconstruction digest differs')
    print(json.dumps(dict(stage='loaded',initialization_seconds=init,input_read_reconstruction_seconds=read_time,archive=raw_path.name)),flush=True)
    profile=cProfile.Profile(); start=time.perf_counter();profile.enable()
    arrays,diagnostics=evaluate(baseline,raw,archive,inputs)
    profile.disable();elapsed=time.perf_counter()-start
    comparison=arrays_check(arrays,raw)
    dcheck={k:array_check(v,receipt['diagnostics'][k]) for k,v in diagnostics.items() if not isinstance(v,str)}
    comparison['diagnostics_exact']=diagnostics==receipt['diagnostics']
    out=HERE/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    profile.dump_stats(str(out.with_suffix('.prof')))
    with out.with_suffix('.txt').open('w') as f:
        pstats.Stats(profile,stream=f).strip_dirs().sort_stats('cumulative').print_stats(55)
    import psutil,scipy
    write_json(out,dict(status='BASELINE_EXACT' if comparison['exact'] and comparison['diagnostics_exact'] else 'BASELINE_MISMATCH',
                       raw=args.raw,raw_sha256=sha256(raw_path),receipt_sha256=sha256(Path(args.reference)/'receipts'/Path(args.raw).with_suffix('.json')),
                       initialization_seconds=init,read_reconstruction_seconds=read_time,evaluation_seconds=elapsed,
                       comparison=comparison,diagnostics=diagnostics,memory=psutil.Process().memory_info()._asdict(),available_memory=psutil.virtual_memory().available,
                       runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),source_pins=PINS))
    print(json.dumps(dict(stage='complete',exact=comparison['exact'],diagnostics_exact=comparison['diagnostics_exact'],seconds=elapsed,output=str(out))),flush=True)

if __name__=='__main__': main()
