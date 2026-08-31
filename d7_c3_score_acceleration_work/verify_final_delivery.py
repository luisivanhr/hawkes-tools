"""Verify repaired source lineage and all linked, preserved certification evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from replay_support import *

def main():
    freeze=read_json(HERE/'CANDIDATE_FREEZE.json');repair=read_json(HERE/'REPAIR_001.json');lock=read_json(HERE/'FINAL_CANDIDATE_LOCK.json');failures=[]
    if sha256(HERE/'REPAIR_001.json')!=lock['repair_receipt_sha256']:failures.append('repair_receipt')
    if sha256(HERE/'CANDIDATE_FREEZE.json')!=lock['original_freeze_sha256']:failures.append('original_freeze')
    for name,expected in freeze['source_sha256'].items():
        target=HERE/repair['before_path'] if name==repair['changed_runtime_source'] else HERE/name
        if sha256(target)!=expected:failures.append('original_source:'+name)
    for name,expected in lock['modules'].items():
        if sha256(HERE/name)!=expected:failures.append('final_source:'+name)
    before=(HERE/repair['before_path']).read_text();after=(HERE/repair['changed_runtime_source']).read_text()
    expected=before.replace('import math\n','import math\nimport warnings\n',1).replace(
        "    if not supported or any(mode not in ('ignore', 'warn') for mode in np.geterr().values()):\n",
        "    if (not supported or any(mode not in ('ignore', 'warn') for mode in np.geterr().values())\n            or any(rule[0] == 'error' for rule in warnings.filters)):\n",1)
    if after!=expected:failures.append('dispatch_only_repair')
    for label,row in repair['evidence'].items():
        if sha256(HERE/row['path'])!=row['sha256']:failures.append('evidence:'+label)
    for name,expected in freeze['prereplay_evidence_sha256'].items():
        if sha256(HERE/'evidence'/name)!=expected:failures.append('prereplay:'+name)
    warning=read_json(HERE/repair['evidence']['warning_policy']['path'])
    if warning['status']!='FOCUSED_REPAIR_EXACT' or warning['default_runtime']['context_aware_warnings']!=0 or warning['default_runtime']['error_filters_present']:failures.append('warning_runtime_proof')
    print(json.dumps(dict(status='FINAL_DELIVERY_VERIFIED' if not failures else 'FAIL',failures=failures,final_modules=lock['modules'],
                          complete_replay_count=1,complete_replay_under_repaired_hash=False)),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
