"""World checkpoint evaluation with full targets, failure objects and immutable NPZ."""
import argparse,hashlib,io,json,subprocess
from pathlib import Path
import numpy as np
import torch
from tools.structured_world.runtime import load_baseline,load_dataset,seed_all
from tools.structured_world_v1p1.budget import start,record
from tools.structured_world_v1p1.reaudit_metrics import write_csv,summarize
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.metrics import diagnostics,match_geometry,prediction_filters


def main():
 p=argparse.ArgumentParser()
 for k in ['trained','manifest','targets','output','ledger','run-id']:p.add_argument('--'+k,required=True)
 p.add_argument('--limit',type=int);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
 blob=Path(a.trained).read_bytes();saved=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);cfg=saved['identity']['config'];args=saved['identity']['arguments'];identity={'arguments':vars(a),'checkpoint_sha256':hashlib.sha256(blob).hexdigest(),'trained_step':saved['step'],'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()}
 start(a.ledger,a.run_id,0,identity)
 try:
  seed_all(42);agent=load_baseline(args['checkpoint'],args['vlm']);agent.model.requires_grad_(False);policy=StructuredWorldPolicy(agent.model,cfg).cuda().eval().requires_grad_(False)
  for name in ['reader','heads']:getattr(policy,name).load_state_dict(saved[name],strict=True)
  ds=load_dataset(agent,a.manifest,args['data_root'],a.limit);rows=[];objects=[];fingerprint=hashlib.sha256()
  for i in range(len(ds)):
   if not record(a.ledger,a.run_id,0):raise RuntimeError('GPU hour cap')
   raw=ds[i];e={k:raw[k] for k in ['image','lang','state','token']};token=e['token']
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):_,batch=policy.encode_conditions([e],world_only=True)
   pred={k:v[0].float() for k,v in batch.items()};np.savez(out/'predictions'/(token+'.npz'),**{k:v.cpu().numpy() for k,v in pred.items()})
   # Targets are opened only AFTER prediction. Require full GT for denominators.
   targetfile=Path(a.targets)/'targets'/(token+'.pt');data=targetfile.read_bytes();fingerprint.update(token.encode()+hashlib.sha256(data).digest());t=WorldTargets(**torch.load(io.BytesIO(data),map_location='cuda',weights_only=True))
   if t.overflow:raise ValueError('Use independently generated full-GT target cache for evaluation')
   row={'token':token,'status':'ok'};row.update(diagnostics(pred,t,include_legacy=False));rows.append(row)
   roi,support,obj=prediction_filters(pred,t);gt_ids=torch.where(t.current_supervision_mask)[0];ids=torch.where(support&obj)[0]
   pr,gc=match_geometry(pred['boxes'][ids,:2],t.current_boxes[gt_ids,:2]);assignment={int(gt_ids[c]):int(ids[r]) for r,c in zip(pr,gc)}
   for g in gt_ids.tolist():
    slot=assignment.get(g);objects.append({'token':token,'track_id':t.track_ids[g],'gt_class':int(t.current_classes[g]),'gt_x':float(t.current_boxes[g,0]),'gt_y':float(t.current_boxes[g,1]),'matched_slot':slot,'detected':slot is not None,'class_correct':None if slot is None else int(pred['logits'][slot].argmax())==int(t.current_classes[g]),'centre_error_m':None if slot is None else float((pred['boxes'][slot,:2]-t.current_boxes[g,:2]).norm()),'future_valid_points':int(t.future_valid_mask[g].sum())})
  write_csv(out/'scenes.csv',rows);write_csv(out/'objects.csv',objects);result=summarize(rows);result.update(identity=identity,targets_manifest_sha256=fingerprint.hexdigest());(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'scenes':len(rows),'recall':result['filtered']['recall_2m'],'precision':result['filtered']['precision_2m'],'class_recall':result['class_filtered']['recall_2m']}),flush=True);record(a.ledger,a.run_id,0,'complete')
 except BaseException:
  record(a.ledger,a.run_id,0,'failed');raise
if __name__=='__main__':main()
