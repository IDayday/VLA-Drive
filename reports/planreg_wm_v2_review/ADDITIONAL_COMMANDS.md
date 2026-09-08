# Additional final-snapshot real teacher command

Executed at `1e01bcbd9e0d2baf668a103ea969ae5e5d980c64` after the main regression, using the final checkpoint. This is an actual RGB/full-model check, not included in the 121 pytest count. Comparison was declared exact before execution; the target and every loss difference were zero.

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=/mnt/project/DriveVLA-M0-planreg-wm-v2-review-20260908:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUPLAN_MAPS_ROOT=/mnt/navsim/maps CUBLAS_WORKSPACE_CONFIG=:4096:8 PLANREG_BASE_VLM_PATH=/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned PLANREG_VQA_VLM_PATH=/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-driving-vqa-dense PLANREG_V2_NORMALIZER=/mnt/project/DriveVLA-M0-stage2/planreg_v2_review_20260908/input_train48/normalizer.json PLANREG_V2_SHARED_INIT=/mnt/project/DriveVLA-M0-stage2/planreg_v2_review_20260908/shared_init_std02_seed0.pt CUDA_VISIBLE_DEVICES=0 /mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python - <<'PY' > /mnt/project/DriveVLA-M0-stage2/planreg_v2_review_20260908/logs/final_teacher_batch_parity.log 2>&1
import json,gc
from pathlib import Path
import torch
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent
from navsim.agents.EpisodeDrive.planreg_v2.data import InputOnlyV2Dataset,v2_collate
from navsim.agents.EpisodeDrive.planreg_v2.losses import world_model_loss
from navsim.agents.EpisodeDrive.planreg_v2.motion import HORIZONS
from navsim.agents.EpisodeDrive.planreg_v2.runtime import source_fingerprint
r=Path('/mnt/project/DriveVLA-M0-stage2/planreg_v2_review_20260908')
# Declared in advance: exact equality, no dynamic tolerance changes.
ckpt=torch.load(r/'final_smoke32/last.ckpt',map_location='cpu',weights_only=False)
cfg=dict(ckpt['config'],shared_init_path=None)
agent=PlanRegV2Agent(cfg,'cuda');agent.load_state_dict(ckpt['model'],strict=True);agent.eval()
del ckpt;gc.collect()
data=InputOnlyV2Dataset(r/'input_train48/manifest.json',cfg['vlm_path']);features,targets=v2_collate([data[0]])
with torch.no_grad():
 p=agent(features)
 old=agent.encode_teacher(features,p,include_current=True)
 new=agent.encode_teacher(features,p)
 targets={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in targets.items()}
 motion=agent.motion_normalizer(targets['motion_sequence'])
 actions,coverage=agent.wm_predictor.motion_encoder(motion,targets['motion_timestamps'],targets['motion_valid'],HORIZONS)
 common=(p['tile_geometry'],p['scene_valid_mask'],p['semantic_queries'].float())
 losses=[]
 for teacher in (old[:,1:],new):
  tf,ro,_=agent.wm_predictor.branches(p['visual_content'].float(),teacher.float(),actions,*common)
  losses.append(world_model_loss(tf,ro,teacher,p['scene_valid_mask'],targets['future_valid_mask'],coverage))
 report=dict(status='PASS' if torch.equal(old[:,1:],new) and all(torch.equal(losses[0][k],losses[1][k]) for k in losses[0]) else 'FAIL',
  input_type='real InternVL3-2B checkpoint/current RGB/three real future frames',production='PlanRegV2Agent.encode_teacher(include_current=True/False) + actual TF/RO loss',
  source_fingerprint=source_fingerprint(),tested_code_commit='1e01bcbd9e0d2baf668a103ea969ae5e5d980c64',
  token=data.records[0]['token'],teacher_future_targets_bitwise_equal=torch.equal(old[:,1:],new),
  declared_atol=0.,declared_rtol=0.,target_max_abs_diff=float((old[:,1:]-new).abs().max()),
  loss_differences={k:float((losses[0][k]-losses[1][k]).abs()) for k in losses[0]},
  current_teacher_student_feature_rms_distance=float((old[:,0].float()-p['visual_content'].float()).square().mean().sqrt()),
  current_teacher_used_only_in_this_diagnostic=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
(r/'TEACHER_BATCH_PARITY.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in report.items() if k!='source_fingerprint'},indent=2))
if report['status']!='PASS':raise AssertionError('Real teacher batch parity failed; do not relabel or loosen tolerance')
PY
```

Report collection first runs `scripts/report_planreg_v2_review.py`. Its conservative pre-profile blocker is then replaced from the executed `GB128_PROFILE.json` and validated layout lock; the final artifact remains `READY_FOR_FORMAL_TRAINING=false` because full-data artifacts and tile-envelope coverage are missing. This enrichment modifies only report JSON, not production/test/config hashes. Raw profile reports and resolved configs are included under `profiles/` for independent recomputation.
