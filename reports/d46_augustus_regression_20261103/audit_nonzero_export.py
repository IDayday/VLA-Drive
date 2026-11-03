"""Independent nonzero native actor versus deployment-loader parameter audit.

CPU only; no scoring, CUDA, evaluation labels or optimizer mutation.
"""
import gc,hashlib,json,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from starVLA.rl.d44_native_rl.parent import load
from starVLA.rl.d44_native_rl.config import source_config
from starVLA.rl.d46_augustus.actor import AugustusActor
from starVLA.rl.d46_augustus.deployment import load_policy
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha


def main():
    torch.set_num_threads(2)
    available=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))*1024
    if available<64*2**30:raise RuntimeError('Defer nonzero CPU audit until64GiB available')
    root=Path('/mnt/project/d46-priority-rl-20261103');out=root/'analysis/AUGUSTUS_REGRESSION_20261103'
    records=[];started=time.monotonic()
    for name,u in [('AUG1_FULL',100),('AUG2_LORA',400)]:
        cp=root/'runs'/(name+'_v1')/'checkpoints'/f'update{u:06}_inner0'
        identity=json.loads((cp/'identity.json').read_text());cfg=identity['recipe']
        model,info=load(cfg['parent_checkpoint'],'cpu',binding=cfg['parent_registration'])
        actor=AugustusActor(model,source_config(cfg),cfg)
        common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
        parameters=dict(actor.named_parameters());adapters=common['adapters']
        assert set(adapters)==set(identity['scope']['trainable_names'])
        changes={};delta_sq=0.;n=0
        with torch.no_grad():
            for k,v in adapters.items():
                target=parameters[k];delta=(v-target).float()
                role='text'if '.language_model.'in k else'action_time_interfaces'
                row=changes.setdefault(role,dict(n=0,squared_delta=0.,max_abs=0.))
                row['n']+=v.numel();row['squared_delta']+=float(delta.square().sum());row['max_abs']=max(row['max_abs'],float(delta.abs().max()))
                n+=v.numel();delta_sq+=float(delta.square().sum());target.copy_(v)
        del common,adapters;gc.collect()
        deployed,_,_,_=load_policy(cp,'cpu')
        a=actor.action_expert.state_dict();b=deployed.state_dict();assert a.keys()==b.keys()
        mismatches=[];digest=hashlib.sha256();checked=0
        for k,v in a.items():
            if not torch.equal(v,b[k]):mismatches.append(k)
            digest.update(k.encode());digest.update(v.detach().contiguous().numpy().tobytes());checked+=v.numel()
        assert not mismatches
        for r in changes.values():r['RMS_delta']=(r['squared_delta']/r['n'])**.5
        record=dict(experiment=name,updates=u,status='PASS',device='CPU',nonzero_trained_weight_delta=delta_sq>0,
            all_native_vs_deployed_tensors_exact=True,checked_numel=checked,trainable_numel=n,roles=changes,
            full_native_deployment_state_sha256=digest.hexdigest(),checkpoint=str(cp),inventory_sha256=file_sha(cp/'files.json'),
            limits='Exact complete parameter/buffer equality and identical deployment model class; no new GPU forward or PDMS score')
        assert delta_sq>0;records.append(record)
        del a,b,actor,model,deployed,parameters;gc.collect()
    result=dict(schema='d46_actual_nonzero_native_deployment_tensor_audit_v1',status='PASS',records=records,seconds=time.monotonic()-started)
    result['sha256']=fingerprint(result);(out/'NONZERO_NATIVE_DEPLOYMENT_AUDIT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='PASS',routes=len(records),seconds=result['seconds'])))


if __name__=='__main__':main()
