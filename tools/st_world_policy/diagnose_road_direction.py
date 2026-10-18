"""Prepare the fixed 32-TRAIN road direction diagnostic for independent review.

This entry is intentionally preparation-only: it reads the already committed
small evidence document, never a rollout, metric cache, model, or score service.
Real execution requires a separate Lead admission and independently reviewed
invocation. Geometry APIs are in flow_posttraining.road_margin_proxy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / 'reports/st_world_policy/20261018/consequence_proxy_evidence_v1/NEXT_DIRECTION_DIAGNOSTIC.json'
SELECTION_HASH = '495547e8f320a975fd31bc5652ee9d372fc03315cd70abd112190f2971703030'
SOURCE_RUNTIME_SHA = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
OWNED = ('starVLA/model/modules/flow_posttraining/road_margin_proxy.py',
         'tools/st_world_policy/diagnose_road_direction.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(evidence_path):
    from starVLA.model.modules.flow_posttraining.road_margin_proxy import RoadDiagnosticSpec
    p = Path(evidence_path)
    if p.stat().st_size > 256*1024:
        raise ValueError('Evidence metadata exceeds preparation read budget')
    raw = p.read_bytes(); evidence = json.loads(raw)
    scenes = evidence['selected']
    if (evidence.get('schema') != 'road_command_direction_diagnostic_feasibility_v1'
            or evidence['source_runtime_commit'] != SOURCE_RUNTIME_SHA
            or evidence['selection_sha256_before_scores'] != SELECTION_HASH
            or evidence['selection_count'] != 32 or len(scenes) != 32
            or len({r['scene_id'] for r in scenes}) != 32
            or len({r['source_log'] for r in scenes}) != 31
            or evidence['policy']['optimizer_step'] != 4095
            or any(r['role'] != 'train' or r['candidate'] != 0 for r in scenes)
            or [(r['rank'],r['slot']) for r in scenes] != [(r,s) for r in range(4) for s in range(8)]):
        raise ValueError('Foreign selection; this entry does not select scenes or change the diagnostic')
    original_keys = ('rank','slot','candidate','scene_id','candidate_id','observation_hash','condition_hash','raw_sha256')
    selected = [{k:r[k] for k in original_keys} for r in scenes]
    actual = digest(json.dumps(selected,sort_keys=True,separators=(',',':')).encode())
    if actual != SELECTION_HASH: raise ValueError('Selection content hash mismatch')
    spec = RoadDiagnosticSpec(); spec.validate()
    return {'schema':'road_direction_preparation_v1','status':'PREPARED_NOT_EXECUTED',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'evidence':{'path':str(p.resolve()),'sha256':digest(raw)},
        'implementation':{name:digest((ROOT/name).read_bytes()) for name in OWNED},
        'selection_hash':SELECTION_HASH,'selected_scene_ids':[r['scene_id'] for r in scenes],
        'scene_count':32,'log_count':31,'policy':evidence['policy'],'constants':asdict(spec),
        'surrogate':'command-space four-corner signed road-union margin; excludes LQR/bicycle gradient',
        'decode_boundary':'stored original GPU physical + float64 original decode delta; independent approval required',
        'canonical_score_backend':'ConsequenceEngine; fixed official reference per candidate; no reward edits',
        'proposed_real_budget':{'cpu_workers':2,'threads_per_worker':1,'wall_seconds':1800,
            'rss_bytes':8*2**30,'output_bytes':2**30,'read_rate_bytes_per_second':32*2**20,
            'max_official_calls_including_retries':128,'max_variants_per_scene':4},
        'real_execution_prerequisites':['independent correctness approval of exact code and anchored decode',
            'independent resource admission and source/software hashes',
            'fresh bounded read budget for rollout and all32 metric-cache schemas/vehicle checks',
            'official interpolation/footprint parity on every actual scene',
            'unchanged fixed32 selection; unresolved scenes retained in all denominators'],
        'existing_DAC':evidence['existing_DAC_counts'],
        'limitation':'All originals DAC=1; no road-repair denominator; no policy-learning conclusion',
        'independent_correctness':'PENDING','independent_resources':'PENDING',
        'training_updates':0,'new_official_calls':0,'gpu_calls':0,'dispatch_authorized':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare'], required=True)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--output', type=Path, required=True, help='New immutable JSON path, at most 2MiB')
    args = parser.parse_args(argv)
    try:
        if args.output.exists(): raise ValueError('Output exists; preserve earlier preparation evidence')
        result = prepare(args.evidence)
        data = (json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        if len(data)>2*2**20: raise ValueError('Output budget exceeded')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as f: f.write(data)
        print(json.dumps({'status':result['status'],'output':str(args.output),'sha256':digest(data),
                          'new_official_calls':0,'dispatch_authorized':False}))
        return 0
    except (ValueError,KeyError,OSError) as error:
        print(json.dumps({'status':'PREPARATION_FAILED','error':str(error)}),file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
