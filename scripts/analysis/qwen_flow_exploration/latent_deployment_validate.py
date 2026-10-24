"""CPU package independence/equivalence on actual learned q weights and tiny source ODE."""
import argparse
import builtins
import copy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import torch
from starVLA.rl.qwen_flow_exploration.protocol import ROOT,read_protocol
from starVLA.rl.qwen_flow_exploration.checkpoint import load_state
from starVLA.rl.qwen_flow_exploration.latent.deployment import export_q_package,load_q_package
from starVLA.rl.qwen_flow_exploration.latent.normalization import policy_from_identity
from starVLA.rl.qwen_flow_exploration.latent.gaussian import sample_roots,joint_log_prob
from starVLA.rl.qwen_flow_exploration.latent import deployment,normalization
from starVLA.rl.qwen_sourceflow.contracts import file_sha,fingerprint
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import seed_for
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    if torch.cuda.is_initialized():raise RuntimeError('CPU package qualification cannot initialize CUDA')
    torch.set_num_threads(2);protocol=read_protocol();checks=[]
    _,first=load_state(next((ROOT/'runs/L1_seed42/checkpoints').glob('round0001_update000000_inner0_*')),protocol['sha256'],restore_rng=False)
    features=first['batch'].features
    torch.manual_seed(781)
    oracle=SourceFlowPolicy(ActionDiT(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,freq_dim=16,
        num_heads=2,attn_head_dim=12,num_layers=3,eps=1e-6,use_gradient_checkpointing=False)).eval().requires_grad_(False)
    condition=dict(context=torch.randn(1,4,40),context_mask=torch.ones(1,4,dtype=torch.bool),visual_mask=torch.ones(1,7,dtype=torch.bool),
                   visual_kv=[dict(k=torch.randn(1,7,24),v=torch.randn(1,7,24)) for _ in range(3)])
    sources=[('L1',ROOT/'runs/L1_seed42',False),
             ('CENTER_qualification',ROOT/'qualification/L1_CENTER_V1_real_model_v1',True)]
    records=[]
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        for name,source,qualification in sources:
            state,payload=load_state(source,protocol['sha256'],restore_rng=False)
            expected=policy_from_identity(payload['identity'],protocol).eval().requires_grad_(False)
            expected.load_state_dict(payload['policy'],strict=True)
            # Delete only disposable source copies, never real experiment assets.
            copyroot=folder/(name+'_synthetic_parent');copyroot.mkdir()
            checkpoint=Path(json.loads((source/'latest.json').read_text())['checkpoint'])
            shutil.copytree(checkpoint,copyroot/'checkpoint')
            if qualification:
                shutil.copy2(payload['identity']['center_normalizer']['path'],copyroot/'old_normalizer.pt')
            package=folder/(name+'_package')
            with patch.object(normalization,'load_center_normalizer',side_effect=AssertionError('Export must use inline state mean')):
                metadata=export_q_package(copyroot/'checkpoint',protocol,package,allow_qualification=qualification)
            shutil.rmtree(copyroot);assert not copyroot.exists()
            # At deployment every actual file read must belong to the package.
            native_open=builtins.open;path_open=Path.open;reads=[]
            def permitted(file):
                if isinstance(file,(str,bytes,Path)):
                    path=Path(file).resolve()
                    if not path.is_relative_to(package.resolve()):raise AssertionError('Forbidden deployment dependency: '+str(path))
                    reads.append(path.name)
            def opened(file,*a,**kw):permitted(file);return native_open(file,*a,**kw)
            def pathopened(path,*a,**kw):permitted(path);return path_open(path,*a,**kw)
            before=torch.get_rng_state().clone()
            with patch.object(builtins,'open',opened),patch.object(Path,'open',pathopened),patch.object(normalization,'load_center_normalizer',side_effect=AssertionError('External normalizer unavailable')):
                loaded,meta=load_q_package(package,protocol,allow_qualification=qualification)
            assert torch.equal(before,torch.get_rng_state())
            assert set(reads)=={'q_weights.pt','identity.json','COMPLETE'} and meta==metadata
            assert tensor_digest(loaded.state_dict())==tensor_digest(expected.state_dict())==metadata['policy_sha256']
            assert all(not p.requires_grad for p in loaded.parameters()) and not loaded.training
            a,b=expected(features),loaded(features)
            assert torch.equal(a.mean,b.mean) and torch.equal(a.log_std,b.log_std)
            byone=[loaded(row[None]) for row in features]
            assert torch.equal(b.mean,torch.cat([p.mean for p in byone])) and torch.equal(b.log_std,torch.cat([p.log_std for p in byone]))
            assert not loaded(features.requires_grad_()).mean.requires_grad;features=features.detach()
            max_ode=0.
            with torch.no_grad():
                for seed in protocol['evaluation_seeds']:
                    ga=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',seed,'CPU_PACKAGE_QUALIFICATION'))
                    gb=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',seed,'CPU_PACKAGE_QUALIFICATION'))
                    qa,qb=expected(features[:1]),loaded(features[:1]);za,zb=sample_roots(qa,1,ga)[:,0],sample_roots(qb,1,gb)[:,0]
                    assert torch.equal(za,zb) and torch.equal(joint_log_prob(qa,za[:,None]),joint_log_prob(qb,zb[:,None]))
                    ta,tb=oracle.sample_ode(condition,za),oracle.sample_ode(condition,zb)
                    max_ode=max(max_ode,float((ta-tb).abs().max()))
                    assert torch.equal(ta,tb) and torch.equal(ActionCodec.denorm_odo(ta),ActionCodec.denorm_odo(tb))
            if qualification:
                try:load_q_package(package,protocol)
                except ValueError:pass
                else:raise AssertionError('Qualification silently became formal')
            records.append(dict(arm=metadata['arm'],qualification=qualification,policy_sha256=metadata['policy_sha256'],
                                package_files=sorted(set(reads)),synthetic_parent_files_removed=True,no_external_asset_reads=True,
                                source_outputs_roots_joint_logprob_B1_equal=True,tiny_source_shift5_10step_ODE_max_abs=max_ode,
                                global_CPU_RNG_unchanged=True,policy_frozen_no_autograd=True))
        checks.append('Actual L1 and actual CENTER qualification export only small q state/inline mean; copied parent checkpoint and normalizer are removed')
        checks.append('Guarded loader reads exactly its3package files; original training/optimizer/normalizer/bank access forbidden; CPU RNG unchanged')
        checks.append('Strict actual source outputs, same-root innovations, joint logprob, B1/B16, and normalized/decoded tiny source10stepODE outputs match exactly')
        checks.append('Deployment policy is frozen/eval without autograd; qualification remains explicitly rejected for formal evaluation')
        # Re-seal a malformed tensor map to exercise strict key checking beyond
        # the ordinary file hash rejection; this is a synthetic corrupt package.
        malformed=folder/'malformed';shutil.copytree(folder/'L1_package',malformed)
        weights=torch.load(malformed/'q_weights.pt',weights_only=True);weights['unexpected_noise_head']=torch.zeros(1)
        atomic_torch(malformed/'q_weights.pt',weights)
        meta=json.loads((malformed/'identity.json').read_text());meta.pop('sha256');meta['policy_sha256']=tensor_digest(weights)
        meta['q_weights_file_sha256']=file_sha(malformed/'q_weights.pt');meta['sha256']=fingerprint(meta);atomic_json(malformed/'identity.json',meta)
        complete=dict(schema=meta['schema'],status='COMPLETE',identity_sha256=meta['sha256'],q_weights_file_sha256=meta['q_weights_file_sha256'])
        complete['sha256']=fingerprint(complete);atomic_json(malformed/'COMPLETE',complete)
        try:load_q_package(malformed,protocol)
        except ValueError:pass
        else:raise AssertionError('Unknown tensor key accepted after seals updated')
        checks.append('Unknown q tensor key is rejected even in a self-consistently re-sealed synthetic package')
    result=dict(schema='pure_q_deployment_CPU_acceptance_v1',status='PASS',qualification_only=True,new_GPU_tasks=0,new_rollouts=0,new_scoring_calls=0,
                checks=checks,cases=records,source_files={str(Path(__file__)):file_sha(__file__),str(Path(deployment.__file__)):file_sha(deployment.__file__)},
                scope='Package/math/tiny sourceODE independence only, not a new driving-score evaluation')
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    atomic_json(out,result);print(json.dumps(dict(status='PASS',checks=len(checks),cases=len(records))))


if __name__=='__main__':main()
