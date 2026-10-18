"""Metadata only: one new matched C1 scale group. Never launches training."""
import json,sys,time,socket,subprocess
from pathlib import Path
ROOT=Path('/mnt/project/st-world-policy-20261016')
REPO=Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
sys.path.insert(0,str(REPO))
from starVLA.dataloader.st_world_targets import atomic_json
from tools.st_world_policy.check_formal_approval import file_hash,source_fingerprint,verify_registration
from tools.st_world_policy.flow_training_config import validate_config,target_contract
from tools.st_world_policy.training_config import environment_record

def artifact(path):return {'path':str(path),'sha256':file_hash(path)}
def read(path):return json.loads(path.read_text())

def main():
 host=socket.gethostname();assert host=='training-vlawm-tsinghua-zt5-worker-0'
 assert host in read(Path('/mnt/project/server_dispatch_policy.json'))['allowed_aliases']
 parent=ROOT/'formal/consequence/registrations/C1_fulltrain_seed42_official_4096_v1'
 assert file_hash(parent/'CONFIG.json')=='f1c4cc851d2e055cc7e2cb10c08e032574c424e4aa8b8f27d46007774a6bba0e'
 name='C1_rollout_std_fulltrain_seed42_official_4096_v1';out=parent.parent/name
 assert not out.exists()
 cfg=read(parent/'CONFIG.json')
 cfg['fpo']['advantage']='scene_center_rollout_std_v1'
 cfg['runtime']['official_execution']['adaptation_reason']='One preregistered C1 rollout-scale variant against completed C1. Same original S0/GT/101592 scenes/seed42/4096 horizon/warm100/world4/global32/micro4/G4/MC50/BC1/policy1. Fresh matched optimizer; no diagnostic weights.'
 validate_config(cfg)
 profile=ROOT/'profiles/scene_center_rollout_std_v1_fulltrain240'
 lease=read(profile/'RESOURCE_LEASE.json')
 lease.update(valid_from_unix=time.time()-1,valid_until_unix=time.time()+72*3600,
              max_invocation_seconds=28800,purpose='Conditional registration only. One4096 endpoint, at most six8h invocations; requires actual240 profile and independent formal evidence/correctness/efficiency approval. No automatic dispatch.')
 out.mkdir()
 for key,value in [('CONFIG.json',cfg),('RESOURCE_LEASE.json',lease),('ENVIRONMENT.json',environment_record()),('TARGET_CONTRACT.json',target_contract(cfg))]:atomic_json(out/key,value,immutable=True)
 reg=read(parent/'REGISTRATION.json');reg.update(run_id=name,source_sha='751fcfa2b41549ae16f7b3204aef8dd18b912016',source_fingerprint=source_fingerprint(REPO))
 reg['implementer_agents']=list(dict.fromkeys(reg['implementer_agents']+['/root/rollout_scale_implementation']))
 for key,name in [('config','CONFIG.json'),('resource_lease','RESOURCE_LEASE.json'),('environment','ENVIRONMENT.json'),('target_contract','TARGET_CONTRACT.json')]:reg['artifacts'][key]=artifact(out/name)
 atomic_json(out/'REGISTRATION.json',reg,immutable=True)
 fingerprint=verify_registration(out/'REGISTRATION.json',repo=REPO)
 atomic_json(out/'PREPARATION.json',{'status':'METADATA_ONLY_NOT_APPROVED_NOT_LAUNCHED','script':artifact(Path(__file__)),
     'fingerprint':fingerprint,'source_of_weights':'original S0 binding unchanged; fresh matched optimizer',
     'diagnostic_weights_used':False,'formal_launch_allowed':False,'required_pending':['complete actual240 profile','independent formal reviews','fresh resource check']},immutable=True)
 print(json.dumps({'status':'METADATA_ONLY_NOT_APPROVED_NOT_LAUNCHED','registration':artifact(out/'REGISTRATION.json')}))

if __name__=='__main__':main()
