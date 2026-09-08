"""Measured production scorer integration, long-target and resolved-LR evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scripts.audit_drivor_scorer_parity import (load_and_verify_upstream_sources,load_upstream_classes,
    _forward_scorer_path,_upstream_pdm_score,DRIVOR_COMMIT)
from navsim.agents.EpisodeDrive.planreg_v2.action import V2ActionDecoder
from navsim.agents.EpisodeDrive.planreg_v2.targets import long_target
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics
from navsim.agents.EpisodeDrive.planreg_v2.runtime import source_fingerprint,load_config
from navsim.agents.EpisodeDrive.planreg_v2.optimizer import build_optimizer


def scorer_integration(device='cpu'):
    sources=load_and_verify_upstream_sources(Path('/mnt/project/external/DrivoR'))
    assert 'tr_out = self.scorer_attention(embedded_traj, scene_features)' in sources['drivor_model.py']
    assert 'tr_out = tr_out+ego_token' in sources['drivor_model.py']
    decoder_cls,heads_cls=load_upstream_classes(sources)
    records=[]
    for seed in (0,13):
        for n in (16,64):
            torch.manual_seed(seed)
            stats=measured_statistics([('unit',torch.randn(8,3),torch.ones(8,dtype=torch.bool))],'train','synthetic-scorer-unit')
            local=V2ActionDecoder(stats).to(device).eval()
            decoder=decoder_cls(4,256,.1,.2,local.config).to(device).eval();heads=heads_cls(local.config).to(device).eval()
            decoder.load_state_dict(local.scorer_attention.state_dict(),strict=True);heads.load_state_dict(local.scorer.state_dict(),strict=True)
            for padding in (False,True):
                p=torch.randn(2,64,8,3,device=device,requires_grad=True)
                memory=torch.randn(2,n,256,device=device,requires_grad=True);ego=torch.randn(2,256,device=device,requires_grad=True)
                valid=None
                if padding:
                    valid=torch.arange(n,device=device)[None]<torch.tensor([n//2,n-3],device=device)[:,None]
                actual,score=local.score(p,memory,valid,ego)
                if padding:
                    rows=[_forward_scorer_path(local.pos_embed,decoder,heads,p[i:i+1],memory[i:i+1,valid[i]],ego[i:i+1,None])[0] for i in range(2)]
                    expected={k:torch.cat([r[k] for r in rows]) for k in rows[0]}
                else:expected=_forward_scorer_path(local.pos_embed,decoder,heads,p,memory,ego[:,None])[0]
                refscore=_upstream_pdm_score(expected,local.config)
                # Exact same kernel/shape gets strict equality. Trimmed reference
                # vs padded batched FP32 attention has a fixed, declared budget.
                atol,rtol=(2e-6,1e-5) if padding else (0.,0.)
                for k in expected:torch.testing.assert_close(actual[k],expected[k],atol=atol,rtol=rtol)
                torch.testing.assert_close(score,refscore,atol=atol,rtol=rtol)
                assert torch.equal(score.argmax(-1),refscore.argmax(-1))
                sum(x.square().mean() for x in actual.values()).backward()
                assert p.grad is None and memory.grad.norm()>0 and ego.grad.norm()>0
                assert all(x.grad is None for x in local.attention.parameters())
                records.append(dict(seed=seed,memory_length=n,padded=padding,atol=atol,rtol=rtol,
                    component_max_abs_diff={k:float((actual[k]-expected[k]).abs().max()) for k in actual},
                    score_max_abs_diff=float((score-refscore).abs().max()),indices_equal=True,
                    proposal_grad_none=True,generator_direct_grad_none=True,scene_grad_norm=float(memory.grad.norm()),ego_grad_norm=float(ego.grad.norm())))
    return dict(status='PASS',kind='actual V2ActionDecoder.score integration, not just component parity',
        upstream_commit=DRIVOR_COMMIT,action_sha256=hashlib.sha256(Path('navsim/agents/EpisodeDrive/planreg_v2/action.py').read_bytes()).hexdigest(),device=device,records=records)


def long_parity():
    old=subprocess.check_output(['git','show','d9ca73f3d61f059285fcbf12a5bc81177ee350d7:navsim/agents/EpisodeDrive/drivevla_features.py'])
    assert b'CubicSpline(x, y)' in old and b'np.cumsum((x_new+1)*alpha)' in old
    q=np.array([1/18,7/6,7/3,32/9,29/6,37/6,68/9,9.])
    rows=[]
    for kind in ('straight','curve','jitter'):
        t=np.arange(11)*.5
        if kind=='jitter':t[1:]+=np.sin(np.arange(1,11))*.009
        p=np.stack([10*t,np.zeros(11),np.zeros(11)],-1)
        if kind!='straight':p=np.stack([2*t+t**2+.02*t**3,np.sin(t*.7)+t**3*.03,t*.12],-1)
        expected=CubicSpline(t[1:],p[1:],bc_type='not-a-knot',extrapolate=False)(np.interp(q,np.arange(10),t[1:]))
        actual,valid=long_target(p,t,np.ones(11,bool))
        np.testing.assert_allclose(actual.numpy(),expected,atol=4e-6,rtol=1e-6)
        assert valid
        rows.append(dict(kind=kind,max_abs_diff=float(np.abs(actual.numpy()-expected).max()),output=actual.tolist()))
    return dict(status='PASS',v1_source_sha256=hashlib.sha256(old).hexdigest(),nominal_query_times=((q+1)*.5).tolist(),records=rows)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--config',help='When supplied, construct actual full agent and audit its actual optimizer groups at GB1/32/128')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    identity=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_fingerprint=source_fingerprint())
    results={'SCORER_INTEGRATION_PARITY':scorer_integration(a.device),'LONG_TARGET_PARITY':long_parity()}
    if a.config:
        from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent
        cfg=load_config(a.config);agent=PlanRegV2Agent(cfg,'cpu')
        rows=[]
        for gb in (1,32,128):
            agent.config['global_batch']=gb
            optimizer,scheduler,groups=build_optimizer(agent,cfg['total_steps'],cfg['learning_rates'])
            rows.append(dict(global_batch=gb,groups=groups,applied_start_lrs=scheduler.get_last_lr()))
        results['LR_RESOLUTION']=dict(status='PASS',input_type='actual full pretrained V2 agent + YAML + AdamW',records=rows)
        r=agent.backbone.planning_register_adapter.planning_registers
        results['INITIALIZATION_AUDIT']=dict(status='PASS',register_std=float(r.std()),register_dtype=str(r.dtype),
            register_shape=list(r.shape),shared_init_path=cfg['shared_init_path'],
            trainable_parameters=sum(p.numel() for p in agent.parameters() if p.requires_grad),
            fp32_trainable=all(p.dtype==torch.float32 for p in agent.parameters() if p.requires_grad))
    for name,result in results.items():
        (root/(name+'.json')).write_text(json.dumps(dict(identity,**result),indent=2))


if __name__=='__main__':main()
