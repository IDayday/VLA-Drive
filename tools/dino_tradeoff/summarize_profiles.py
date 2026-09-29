"""Cost-only P0 summaries; never interpret profile losses as effect results."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.ddpolicy_vehicle.prepare_data import atomic_json

def summarize(run):
 ident=json.loads((run/'identity.json').read_text());rows=[json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()]
 measured=[r for r in rows if 20<r['update']<=120];all_rank=np.asarray([r['per_rank_profile'] for r in measured])
 if not len(measured):raise ValueError('No stable measured steps')
 seconds=all_rank[:,:,0].max(1);world=ident['world_size'];batch=ident['global_batch']
 report={'run':run.name,'arm':ident['config']['foresight']['arm'],'source':ident['source_sha'],'world':world,'global_batch':batch,'microbatch':ident['micro_batch'],'accumulation':batch//world/ident['micro_batch'],'measured_steps':len(measured),'measurement_complete':len(measured)==100,'step_p50_s':float(np.median(seconds)),'step_p95_s':float(np.percentile(seconds,95)),'samples_per_s':float(batch/seconds.mean()),'warm_training_gpu_hours_per_epoch':float(seconds.mean()*world*np.ceil(101592/batch)/3600),'per_rank_peak_allocated_bytes':all_rank[:,:,2].max(0).tolist(),'per_rank_peak_reserved_bytes':all_rank[:,:,3].max(0).tolist(),'sum_peak_allocated_bytes':float(all_rank[:,:,2].max(0).sum()),'data_wait_p50_s':float(np.median(all_rank[:,:,1].max(1))),'total_sequence_minmax':[float(all_rank[:,:,4].min()),float(all_rank[:,:,4].max())],'profile_only_not_formal_result':True}
 if (run/'inference_profile.json').exists():
  infer=json.loads((run/'inference_profile.json').read_text());times=np.asarray(infer['per_rank_seconds']);report.update(batch1_latency_p50_s=float(np.median(times)),batch1_latency_p95_s=float(np.percentile(times,95)))
 if (run/'checkpoint_costs.jsonl').exists():report['checkpoint_costs']=[json.loads(x) for x in (run/'checkpoint_costs.jsonl').read_text().splitlines()]
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 results=[]
 for run in sorted((Path(a.campaign)/'students').glob('*')):
  if (run/'steps.jsonl').exists() and json.loads((run/'identity.json').read_text())['scope']=='profile':
   try:results.append(summarize(run))
   except ValueError:pass
 atomic_json(a.output,{'profiles':results,'selection':'pending full-data paired PDMS; no recommendation from cost alone'})
if __name__=='__main__':main()
