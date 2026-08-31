"""Verify the frozen artifact tree and its explicitly permitted source inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',default='evidence/source_preservation.json');args=p.parse_args()
    reference=Path(args.reference).resolve();common,adapter,baseline=setup(reference)
    manifest=read_json(reference/'COMPLETE_ARTIFACT_MANIFEST.json');failures=[];total=0
    for entry in manifest['entries']:
        path=reference/entry['path']
        if common.forbidden_path(path):raise RuntimeError('unexpected forbidden member in current frozen experiment')
        actual=sha256(path);total+=path.stat().st_size
        if actual!=entry['sha256']:failures.append(entry['path'])
    deps=read_json(reference/'SOURCE_PRESERVATION_RECEIPT.json')['dependencies'];checked={}
    for path,expected in deps.items():
        if common.forbidden_path(path):continue
        actual=sha256(path);checked[path]=actual
        if actual!=expected:failures.append(path)
    result=dict(status='PRESERVED' if not failures else 'DRIFT',frozen_manifest_sha256=sha256(reference/'COMPLETE_ARTIFACT_MANIFEST.json'),
                artifact_count=len(manifest['entries']),artifact_bytes=total,dependency_hashes=checked,failures=failures,source_pins=PINS,
                scope='All entries in the current executed experiment manifest plus explicitly permitted dependency receipt files; no historical truth-code exclusions opened or rehashed.')
    write_json(HERE/args.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('dependency_hashes','source_pins')}),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
