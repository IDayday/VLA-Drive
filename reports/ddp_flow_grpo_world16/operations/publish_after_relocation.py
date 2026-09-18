"""Overlap existing full asset checks with bounded relocation validation.

Uses the same source-bound verifier and native semantic publisher as release.py.
No training is launched; a failed/missing placement comparison never publishes.
"""
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from scripts.cluster_flow_grpo.identity import configure_release, orchestration_identity, placement_evidence
from scripts.cluster_flow_grpo.verify_profile import verify
from scripts.cluster_flow_grpo.test_release import validate_cpu_receipt
from scripts.flow_grpo.publish_acceptance import publish
from starVLA.rl.flow_grpo.acceptance import acceptance_context
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.reproducibility import configure_numerics, resume_assets

plan = json.loads(Path('configs/cluster_flow_grpo/paired_world16.json').read_text())
from scripts.cluster_flow_grpo.assets import install_receipt
reuse = plan['asset_verification_receipt']
install_receipt(reuse['path'], reuse['sha256'])
identity, directory = configure_release()
os.environ.update(WORLD_SIZE='16', FLASH_ATTENTION_DETERMINISTIC='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
configure_numerics()
validate_cpu_receipt(directory/'cpu_validation.json', identity)

def context(short):
    variant = {'f':'frozen_visual', 'u':'unfrozen_visual'}[short]
    cfg, sft = resolve_config(f'configs/flow_grpo/paired_world16_{variant}.yaml')
    result = acceptance_context(cfg, resume_assets(cfg, sft))
    print(variant, 'PRIOR_FULL_ASSET_VERIFICATION_REUSED', flush=True)
    return short, variant, cfg, result

with ThreadPoolExecutor(max_workers=2) as pool:
    checked = list(pool.map(context, ['f','u']))
proof = Path('runs/resource_reallocation_v3/f16_within_layout_resume_comparison.json')
deadline = time.monotonic()+1800
while not proof.exists():
    if time.monotonic()>deadline: raise TimeoutError('bounded relocation proof deadline')
    if orchestration_identity()!=identity: raise ValueError('source changed during verification')
    time.sleep(5)
if json.loads(proof.read_text())['status']!='PASS': raise ValueError('relocation comparison failed; no release')
placement_evidence()
for short, variant, cfg, current in checked:
    marker = json.loads(Path(f'runs/resource_reallocation_v2/{short}16_postprocess_complete.json').read_text())
    if marker.get('status')!='PASS': raise ValueError('original qualification incomplete')
    if orchestration_identity()!=identity: raise ValueError('source changed during verification')
    evidence = directory/(variant+'_evidence.json')
    verify(short, 'runs/resource_reallocation_v2', evidence)
    record = publish(cfg, current, [evidence], cfg['runtime']['acceptance_record'])
    print(json.dumps({'variant':variant,'status':record['status'],'gates':len(record['gates']),
                      'release_directory':str(directory)}), flush=True)
