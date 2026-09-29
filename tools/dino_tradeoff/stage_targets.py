"""Hash-verified, atomic local replicas of immutable target chunks."""
import argparse,json,shutil,os,time
from pathlib import Path
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--candidate',required=True);p.add_argument('--images',type=int,default=0);a=p.parse_args()
 source=Path(a.source)/a.candidate;dest=Path(a.output)/a.candidate;dest.mkdir(parents=True,exist_ok=True)
 ident=json.loads((source/'identity.json').read_text());wanted=min(a.images or ident['image_count'],ident['image_count']);chunks=(wanted+ident['chunk_size']-1)//ident['chunk_size'];start=time.time()
 required=wanted*1024*ident['candidate']['tokens_per_view']*2
 if shutil.disk_usage(dest).free<required+100*2**30:raise RuntimeError('Insufficient local space including100GiB reserve')
 if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=ident:raise ValueError('Foreign local replica')
 atomic_json(dest/'identity.json',ident)
 total=0
 for i in range(chunks):
  name=f'chunk_{i:06d}';meta=json.loads((source/(name+'.json')).read_text());origin=source/(name+'.safetensors');target=dest/origin.name
  if meta['identity']!=ident['identity']:raise ValueError('Source identity changed')
  if not target.exists():
   temp=target.with_suffix('.'+str(os.getpid())+'.tmp');shutil.copy2(origin,temp)
   if file_sha256(temp)!=meta['sha256']:raise ValueError('Copy corruption')
   temp.replace(target)
  elif file_sha256(target)!=meta['sha256']:raise ValueError('Existing local corruption')
  atomic_json(dest/(name+'.json'),meta);total+=meta['bytes']
 if wanted==ident['image_count']:
  complete=json.loads((source/'COMPLETE.json').read_text())
  if complete['identity']!=ident['identity']:raise ValueError('Source incomplete')
  atomic_json(dest/'COMPLETE.json',complete)
 atomic_json(dest/'local_replica.json',{'identity':ident['identity'],'images':wanted,'chunks':chunks,'bytes':total,'seconds':time.time()-start,'verified_sha256':True})
if __name__=='__main__':main()
