"""Copy unchanged inputs/labels to this host's local disk, with byte checks."""
import argparse,json,hashlib,os,shutil,time,socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--image-root',required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
 src=Path(a.source);out=Path(a.output);images=Path(a.image_root);out.mkdir(parents=True,exist_ok=True);images.mkdir(parents=True,exist_ok=True)
 if shutil.disk_usage(out).free<100*2**30:raise RuntimeError('Keep100GiB local free space reserve')
 ident=json.loads((src/'identity.json').read_text());rows=json.loads((src/'index.json').read_text());selected=rows[:a.limit] if a.limit else rows
 def copy_file(source,dest):
  dest.parent.mkdir(parents=True,exist_ok=True)
  sha=file_sha256(source)
  if dest.exists():
   if file_sha256(dest)!=sha:raise ValueError('Local replica changed')
  else:
   temp=dest.with_suffix('.'+str(os.getpid())+'.tmp');shutil.copy2(source,temp);temp.replace(dest)
  return sha
 for name in ('identity.json','index.json','ego_identity.json','COMPLETE.json'):copy_file(src/name,out/name)
 def stage(r):
  token=r['token'];record=json.loads((src/'current'/(token+'.json')).read_text());hashes={}
  copy_file(src/'current'/(token+'.json'),out/'current'/(token+'.json'));copy_file(src/'ego'/(token+'.pt'),out/'ego'/(token+'.pt'))
  for path in record['image_paths']:
   dest=images/(hashlib.sha256(path.encode()).hexdigest()+'.jpg');hashes[path]=copy_file(path,dest)
  return hashes
 started=time.time();hashes={}
 with ThreadPoolExecutor(max_workers=16) as pool:
  for result in pool.map(stage,selected):hashes.update(result)
 atomic_json(out/'local_image_hashes.json',hashes)
 atomic_json(out/'local_stage.json',{'host':socket.gethostname(),'source_identity':ident['identity'],'scenes':len(selected),'total_scenes':len(rows),'images':len(hashes),'image_root':str(images),'image_hash_manifest':file_sha256(out/'local_image_hashes.json'),'seconds':time.time()-started,'source_bytes_unchanged':True})
if __name__=='__main__':main()
