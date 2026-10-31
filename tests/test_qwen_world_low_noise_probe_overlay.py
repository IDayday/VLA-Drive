"""Minimal CPU source-sampler pairing and fixed-commit overlay checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest
import torch

from tools.qwen_world_research.probe_zero_rollouts import (
    configure_diagnostic_noise,paired_chains,single_sde_reusing_baseline,sealed,sha,
)
from tools.qwen_world_research.freeze_overlay import freeze_overlay
from starVLA.rl.simwam_recipe.vendor.action_math import ActionGRPOMixin
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler


class Toy(ActionGRPOMixin):
    def __init__(self):
        self.device=torch.device('cpu');self.torch_dtype=self.grpo_action_dtype=torch.float32
        self.action_expert=SimpleNamespace(action_dim=3)
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=5.)
        self.grpo_num_inference_steps=10;self.grpo_infer_shift=None
        self.grpo_sample_noise_std=self.grpo_logprob_noise_std=self.grpo_noise_level=.1
        self.grpo_sde_mode='rigorous';self.grpo_min_std=.001;self.grpo_randn_clip=5.
        self.grpo_train_sde_steps=3;self.grpo_sde_step_indices=None;self.grpo_sde_exclude_last_step=False
        self.grpo_merge_sde_window=False;self.grpo_final_action_clip=None
        self.grpo_sigma_max_guard=.99

    def _randn(self,shape,device,dtype,generator):
        return torch.randn(shape,generator=generator,dtype=dtype).to(device)

    def action_velocity(self,x,time,cond,use_ref=False):return x*.17+.03


def test_lower_noise_changes_only_rigorous_field_and_preserves_roots_prefix_and_ODE():
    a,b=Toy(),Toy();audit=configure_diagnostic_noise(b,.03)
    assert b.grpo_noise_level==.03 and b.grpo_sample_noise_std==b.grpo_logprob_noise_std==.1
    condition=dict(context=torch.zeros(8,1,1))
    oa,sa,da,aa=paired_chains(a,condition,'cpu-scene')
    ob,sb,db,ab=paired_chains(b,condition,'cpu-scene')
    assert torch.equal(oa,ob) and torch.equal(sa[:,:8],sb[:,:8])
    assert aa==ab and all(torch.equal(x,y) for x,y in zip(da['SDE8'],db['SDE8']))
    assert not torch.equal(sa[:,8:],sb[:,8:])
    x=torch.ones(1,8,3);v=torch.zeros_like(x)
    ma,std_a=a._sde_mean_std(x,v,.4,-.1);mb,std_b=b._sde_mean_std(x,v,.4,-.1)
    assert std_b/std_a==pytest.approx(.3)
    torch.testing.assert_close(1-mb,(1-ma)*.09,atol=6e-8,rtol=0)
    assert audit['active_field']=='grpo_noise_level'
    with pytest.raises(ValueError):configure_diagnostic_noise(Toy(),.07)


def test_overlay_uses_fixed_git_bytes_links_unchanged_tree_and_never_overwrites(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,stdout=subprocess.PIPE).stdout.decode().strip()
    git('init','-q');git('config','user.name','CPU test');git('config','user.email','cpu@example.invalid')
    (repo/'pkg').mkdir();(repo/'pkg/a.py').write_text('base\n');(repo/'pkg/b.py').write_text('unchanged\n')
    (repo/'reports').mkdir();(repo/'reports/history.html').write_text('large historical subtree is not traversed\n')
    git('add','.');git('commit','-qm','base');base_commit=git('rev-parse','HEAD')
    base=tmp_path/'base';(base/'pkg').mkdir(parents=True);(base/'reports').mkdir()
    for name in ('pkg/a.py','pkg/b.py','reports/history.html'):(base/name).write_bytes((repo/name).read_bytes())
    (base/'FROZEN_SOURCE.json').write_text(json.dumps(dict(commit=base_commit)))
    (repo/'pkg/a.py').write_text('committed delta\n');(repo/'pkg/new.py').write_text('new committed\n')
    git('add','.');git('commit','-qm','delta');commit=git('rev-parse','HEAD')
    (repo/'pkg/a.py').write_text('DIRTY MUST NOT FREEZE\n')
    out=tmp_path/'overlay';result=freeze_overlay(repo,base,commit,out)
    assert (out/'pkg/a.py').read_text()=='committed delta\n'
    assert (out/'pkg/new.py').read_text()=='new committed\n'
    assert not (out/'pkg/a.py').is_symlink() and (out/'pkg/b.py').is_symlink()
    assert (out/'reports').is_symlink() and (base/'pkg/a.py').read_text()=='base\n'
    assert result['copied_regular_files']==2 and len(result['expanded_directories'])==2
    meta=json.loads((out/'FROZEN_SOURCE.json').read_text());assert meta['commit']==commit and meta['base']['commit']==base_commit
    with pytest.raises(FileExistsError):freeze_overlay(repo,base,commit,out)


def test_lower03_calls_only_SDE_and_reuses_sealed_ODE_feedback_and_B1(tmp_path):
    import numpy as np
    baseline=Toy();condition=dict(context=torch.zeros(8,1,1));token='cpu-scene'
    ode,sde,draws,audit=paired_chains(baseline,condition,token)
    folder=tmp_path/'baseline';(folder/'scenes').mkdir(parents=True)
    arrays=folder/'scenes'/f'{token}.npz'
    np.savez_compressed(arrays,ODE_chain=ode.numpy(),SDE_chain=sde.numpy(),prior=draws['SDE8'][0].numpy(),
                        innovations=torch.stack(draws['SDE8'][1:]).numpy())
    row=sealed(dict(identity='baseline-identity',pairing=audit,arrays=dict(path='scenes/'+arrays.name,sha256=sha(arrays)),
                    ODE8=[dict(reused_index=i) for i in range(8)],canonical_single_ODE=dict(reused_B1=True)))
    path=folder/'scenes'/f'{token}.json';path.write_text(json.dumps(row))
    manifest=dict(baseline_probe=dict(output=str(folder),identity_sha256='baseline-identity',scene_file_sha256={token:sha(path)}))
    actor=Toy();configure_diagnostic_noise(actor,.03);calls=[];original=actor.sample_action_chain
    def sample(*args,**kwargs):
        calls.append(kwargs.get('deterministic'));return original(*args,**kwargs)
    actor.sample_action_chain=sample
    reused,new,observed,new_audit,cross,old=single_sde_reusing_baseline(actor,condition,manifest,token)
    assert calls==[False] and torch.equal(reused,ode) and not torch.equal(new,sde)
    assert torch.equal(new[:,:8],ode[:,:8]) and new_audit==audit
    assert cross['new_scored_candidates']==8 and cross['reused_ODE_candidates']==8
    assert old['ODE8']==row['ODE8'] and old['canonical_single_ODE']==row['canonical_single_ODE']
