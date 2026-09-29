"""Project verified current-frame identities into a new current-only index."""
import argparse,json
from pathlib import Path
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--source-index',required=True);p.add_argument('--train',required=True);p.add_argument('--dev',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 src=Path(a.source_index);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 ident=json.loads((src/'identity.json').read_text())
 if not (src/'COMPLETE.json').exists():raise ValueError('Source index incomplete')
 if file_sha256(src/'images.json')!=ident['files']['images']:raise ValueError('Source image index changed')
 old=json.loads((src/'images.json').read_text());images=[];lookup={};files={};populations={};logs={};tokens={}
 for split,root in [('train',Path(a.train)),('dev',Path(a.dev))]:
  current=json.loads((root/'identity.json').read_text());index=json.loads((root/'index.json').read_text())
  rows=json.loads((src/(split+'_scenes.json')).read_text())
  if file_sha256(src/(split+'_scenes.json'))!=ident['files'][split] or identity_hash(index)!=ident['index_hashes'][split] or current['partition_sha256']!=ident['partition_sha256']:raise ValueError('Source split changed')
  if [r['token'] for r in index]!=[r['token'] for r in rows]:raise ValueError('Scene order changed')
  new=[];logs[split]={r['log'] for r in index};tokens[split]={r['token'] for r in index}
  for i,r in enumerate(rows):
   frame=json.loads((root/'current'/(r['token']+'.json')).read_text());ids=[]
   for view,oldid in enumerate(r['images'][0]):
    entry=old[oldid]
    if oldid<0 or entry['path']!=frame['image_paths'][view] or entry['timestamp']!=frame['timestamp'] or entry['view']!=view or entry['split']!=split:raise ValueError('h0 differs from actual student decision image')
    if oldid not in lookup:lookup[oldid]=len(images);images.append(entry)
    ids.append(lookup[oldid])
   new.append({'token':r['token'],'log':index[i]['log'],'images':ids})
  atomic_json(out/(split+'_scenes.json'),new);files[split]=file_sha256(out/(split+'_scenes.json'));populations[split]={'scenes':len(new),'logs':len(logs[split])}
 if logs['train']&logs['dev'] or tokens['train']&tokens['dev']:raise ValueError('Split leakage')
 atomic_json(out/'images.json',images);files['images']=file_sha256(out/'images.json')
 record={'schema':'dino_tradeoff_current_index_v1','source_index':ident['identity'],'index_hashes':ident['index_hashes'],'partition_sha256':ident['partition_sha256'],'files':files,'population':populations,'image_count':len(images),'raw_image_bytes':sum(i['bytes'] for i in images),'horizon_s':0.,'h0_checked_against_every_student_record':True}
 record['identity']=identity_hash(record);atomic_json(out/'identity.json',record);atomic_json(out/'COMPLETE.json',record)
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
