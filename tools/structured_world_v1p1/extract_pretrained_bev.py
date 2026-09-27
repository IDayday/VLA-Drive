"""Rebuildable current-image-only DAV2 + calibrated BEV cache, no target loading."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import torch
from tools.structured_world.provider_cli import current_observation
from tools.structured_world_v1p1.budget import start,record
from starVLA.model.modules.structured_world.pretrained_bev import PretrainedVisualBEVProvider
from starVLA.model.modules.structured_world.providers import calibration_fingerprint


def main():
 p=argparse.ArgumentParser()
 for key in ['weights','weight-sha256','manifest','observations','sensor-root','output','ledger','run-id']:p.add_argument('--'+key,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 identity={'arguments':vars(a),'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()}
 start(a.ledger,a.run_id,0,identity)
 try:
  model=PretrainedVisualBEVProvider(a.weights,a.weight_sha256).cuda().eval();tokens=json.loads(Path(a.manifest).read_text());rows=[]
  (out/'manifest.json').write_text(json.dumps(identity,indent=2))
  for token in tokens:
   if not record(a.ledger,a.run_id,0):raise RuntimeError('GPU hour cap reached')
   inputs,digests=current_observation(Path(a.observations)/(token+'.npz'),a.sensor_root,'cuda')
   torch.cuda.synchronize();t=time.perf_counter()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f,xyz,support,meta=model(inputs)
   torch.cuda.synchronize();latency=time.perf_counter()-t
   if not torch.isfinite(f).all():raise ValueError('Nonfinite pretrained BEV')
   meta.update(scene_token=token,decision_time=int(inputs.decision_time[0]),image_sha256=digests,
               image_transforms=inputs.image_transforms.cpu().tolist(),calibration_sha256=calibration_fingerprint(inputs),dtype=str(f.dtype),provider_code_sha=identity['code_sha'])
   torch.save({'features':f.cpu(),'coordinates':xyz.cpu(),'observation_support':support.cpu(),'metadata':meta},out/(token+'.pt'))
   rows.append({'token':token,'latency_seconds':latency,'supported_cells':int(support.sum()),'feature_shape':list(f.shape)})
   if len(rows)%16==0:print(json.dumps({'completed':len(rows),'last':rows[-1]}),flush=True)
  (out/'extraction.json').write_text(json.dumps({'records':rows,'failed':0,'provider_metadata':model.metadata(),'excluded_original_depth_head_keys':model.encoder.excluded_depth_head_keys},indent=2))
  record(a.ledger,a.run_id,0,'complete')
 except BaseException:
  record(a.ledger,a.run_id,0,'failed');raise
if __name__=='__main__':main()
