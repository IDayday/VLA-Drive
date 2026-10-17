"""Mechanical narrow v4 -> v5 candidate generation; never deploys/runs helpers."""
import difflib
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'W_LAUNCH_HELPER_v4'
PREP=HERE.parent/'full_dev_C4_execution_v1'
ROOT=Path('/mnt/project/st-world-policy-20261016')
pins={n:hashlib.sha256((PREP/n).read_bytes()).hexdigest() for n in ('POLICY.json','INTENTS.json','COMMANDS.json','TRAINING_BINDINGS.json','EXECUTION_PLAN.json')}
addition="\n# Only C4 is added; all older preparation pins and routes remain fixed.\nC4_V5_PREP = Path("+repr(str(PREP))+")\nC4_V5_PINS = "+repr(pins)+"\n\n\ndef c4_pin_refs_v5():\n    return {name: {'path': str(C4_V5_PREP / name), 'sha256': value}\n            for name, value in C4_V5_PINS.items()}\n\n"
changes=[]
for stem in ('launch_formal_dev_export','launch_completed_dev_score'):
    old=OLD/(stem+'_v4.py'); original=old.read_text()
    assert (ROOT/'resources'/old.name).read_bytes()==old.read_bytes()
    candidate=original.replace('\n# Fixed host routing for this campaign;',addition+'\n# Fixed host routing for this campaign;',1)
    candidate=candidate.replace('Only the two explicit source-pinned C2/C3 preparations are accepted.','Only the explicit source-pinned C2/C3/C4 preparations are accepted.')
    candidate=candidate.replace("assert arm in ('C2', 'C3')", "assert arm in ('C2', 'C3', 'C4')")
    candidate=candidate.replace("(additive_pin_refs_v2() if arm == 'C2' else c3_pin_refs_v3())", "(additive_pin_refs_v2() if arm == 'C2' else c3_pin_refs_v3() if arm == 'C3' else c4_pin_refs_v5())")
    candidate=candidate.replace("def load_fixed_additive_v3(job_id, base_documents):\n", "def load_fixed_additive_v3(job_id, base_documents):\n    if job_id.startswith('C4_'):\n        return load_pinned_additive_v3(job_id, base_documents, 'C4', c4_pin_refs_v5())\n")
    candidate=candidate.replace("if intent['arm'] in ('C2', 'C3'):", "if intent['arm'] in ('C2', 'C3', 'C4'):")
    candidate=candidate.replace("(additive_pin_refs_v2() if intent['arm'] == 'C2' else c3_pin_refs_v3())", "(additive_pin_refs_v2() if intent['arm'] == 'C2' else c3_pin_refs_v3() if intent['arm'] == 'C3' else c4_pin_refs_v5())")
    target=HERE/(stem+'_v5.py')
    with target.open('x') as f:f.write(candidate)
    diff=''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile=old.name,tofile=target.name))
    with (HERE/(stem+'_v5.diff')).open('x') as f:f.write(diff)
    changes.append(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),parent_path=str(ROOT/'resources'/old.name),parent_sha256=hashlib.sha256(old.read_bytes()).hexdigest()))
with (HERE/'CANDIDATES.json').open('x') as f:json.dump(dict(status='CANDIDATE_PENDING_INDEPENDENT_REVIEW_AND_LEAD_DEPLOYMENT',base_source='abd51fd0893804e1f04d9a52f9188d5669b33296',candidates=changes,c4_preparation=str(PREP),c4_pins=pins,deployed=False,launch_authorized=False),f,indent=2);f.write('\n')
