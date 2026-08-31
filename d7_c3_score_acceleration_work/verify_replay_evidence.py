"""Strict post-replay reconciliation, including scalar diagnostic bit patterns.

This reads recomputed evidence, never bypasses or reruns the changed evaluator.
It also proves that every pre-replay frozen Python source stayed unchanged.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def exact_json(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(exact_json(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact_json(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return struct.pack('d',a)==struct.pack('d',b)
    return a==b

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--replay',default='evidence/full_replay');p.add_argument('--output',default='evidence/replay_reconciliation.json');args=p.parse_args()
    reference=Path(args.reference).resolve();replay=(HERE/args.replay).resolve();freeze=read_json(HERE/'CANDIDATE_FREEZE.json');failures=[]
    for name,expected in freeze['source_sha256'].items():
        if sha256(HERE/name)!=expected:failures.append('frozen_source:'+name)
    if sha256(HERE/'EXACTNESS_CONTRACT.md')!=freeze['contract_sha256']:failures.append('exactness_contract')
    for name,expected in freeze['prereplay_evidence_sha256'].items():
        if sha256(HERE/'evidence'/name)!=expected:failures.append('prereplay_evidence:'+name)
    manifest={x['path']:x['sha256'] for x in read_json(reference/'COMPLETE_ARTIFACT_MANIFEST.json')['entries']}
    seen=set();trace_digests=set();seeds=set();raw_arrays=0;node_arrays=0;node_rows=[];input_rows=[]
    for path in sorted((replay/'node_evidence').glob('*.json')):
        node=read_json(path);node_rows.append(dict(path=str(path.relative_to(HERE)),sha256=sha256(path)))
        if not node['exact']:failures.append('node:'+node['node_id'])
        for level in node['levels']:
            node_arrays+=len(level['comparison']['arrays'])
            if not level['comparison']['exact'] or not all(level['summary_checks'].values()):failures.append('level:'+node['node_id'])
        for job in node['jobs']:
            name=job['raw'];rel='raw/'+name
            if name in seen:failures.append('duplicate:'+name)
            seen.add(name);trace_digests.add(job['trace_archive_sha256']);seeds.add(job['seed'])
            if job['raw_sha256']!=manifest[rel]:failures.append('rawpin:'+name)
            receipt_path=reference/'receipts'/Path(name).with_suffix('.json')
            if sha256(receipt_path)!=manifest['receipts/'+receipt_path.name]:failures.append('receiptpin:'+name)
            receipt=read_json(receipt_path)
            if not exact_json(job['fresh_diagnostics'],receipt['diagnostics']):failures.append('diagnostic_bits:'+name)
            if not job['comparison']['exact']:failures.append('array:'+name)
            raw_arrays+=len(job['comparison']['arrays'])
            input_rows.append(dict(raw=rel,sha256=job['raw_sha256'],seed=job['seed'],trace_archive_sha256=job['trace_archive_sha256']))
    expected_names={Path(x).name for x in manifest if x.startswith('raw/') and x.endswith('.npz')}
    if seen!=expected_names:failures.append('archive_set')
    if (len(node_rows),len(seen),len(seeds),len(trace_digests))!=(39,624,624,624):failures.append('counts')
    report=read_json(replay/'EXACTNESS_REPLAY.json')
    if report['status']!='RAW_AND_NODE_REPLAY_EXACT':failures.append('replay_status')
    output=dict(status='EXACT_REPLAY_RECONCILED' if not failures else 'FAIL',failures=failures,raw_archive_count=len(seen),
                unique_seeds=len(seeds),unique_trace_digests=len(trace_digests),raw_array_byte_checks=raw_arrays,node_array_byte_checks=node_arrays,
                scalar_diagnostics_byte_checks=624,frozen_code_unchanged=True if not any(x.startswith('frozen_source') for x in failures) else False,
                candidate_freeze_sha256=sha256(HERE/'CANDIDATE_FREEZE.json'),full_replay_report_sha256=sha256(replay/'EXACTNESS_REPLAY.json'),
                node_evidence_manifest=node_rows,inputs=sorted(input_rows,key=lambda x:x['raw']),source_sha256=sha256(__file__))
    write_json(HERE/args.output,output);print(json.dumps({k:v for k,v in output.items() if k not in ('inputs','node_evidence_manifest')}),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
