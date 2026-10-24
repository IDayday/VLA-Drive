"""Shared read-only single-output inference and independent scene partitioning."""
import json
from pathlib import Path
import subprocess
import torch
from .native import NativeEnvironment
from .latent.policy import condition_features
from .latent.gaussian import sample_roots,joint_kl,joint_entropy
from .preservation.deployment import read_adapter_package,merge_native_adapters_
from .common_analysis import checked_seal,validate_root_record
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.simwam_recipe.contracts import seed_for


SCHEMA='three_route_partitioned_single_deployment_v1'
ROOT_DESCRIPTION='native_qwen_sourceflow_ODE scene-bound CPU FP32 namespace'


def contract(protocol):
    return dict(root_namespace='native_qwen_sourceflow_ODE',root_rng_device='cpu',
                root_innovation_dtype='float32',steps=10,time_grid='source_shift5',
                horizon=8,action_dim=3,base_model_sha256=protocol['base_model_sha256'])


def partition(tokens,rank,world):
    if not 1<=world<=64 or not 0<=rank<world or world>len(tokens) or len(set(tokens))!=len(tokens):
        raise ValueError('Invalid or empty independent scene partition')
    return tokens[rank::world]


def source_files():
    base=Path(__file__).resolve().parents[3]
    names=['starVLA/rl/qwen_flow_exploration/evaluation.py',
           'scripts/analysis/qwen_flow_exploration/evaluate_sharded.py',
           'starVLA/rl/qwen_flow_exploration/native.py',
           'starVLA/rl/qwen_flow_exploration/protocol.py',
           'starVLA/rl/qwen_flow_exploration/evaluation_admission.py',
           'starVLA/rl/qwen_flow_exploration/latent/deployment.py',
           'starVLA/rl/qwen_flow_exploration/latent/policy.py',
           'starVLA/rl/qwen_flow_exploration/latent/gaussian.py',
           'starVLA/rl/qwen_flow_exploration/preservation/deployment.py',
           'starVLA/model/modules/qwen_sourceflow/source_flow_policy.py',
           'starVLA/model/modules/qwen_sourceflow/cached_action_runner.py',
           'starVLA/model/modules/qwen_sourceflow/qwen_encoder.py',
           'starVLA/model/modules/qwen_sourceflow/condition_provider.py',
           'starVLA/model/modules/qwen_sourceflow/action_codec.py']
    files={name:file_sha(base/name) for name in names}
    files['__git_commit__']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip()
    return files


def qualified_execution(path,qualification):
    if qualification:return None
    if not path:raise ValueError('Formal partitioned evaluation requires executed equivalence evidence')
    if subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[3],text=True).strip():
        raise ValueError('Formal partitioned evaluation requires a clean immutable source')
    proof=checked_seal(json.loads(Path(path).read_text()),'partitioned inference qualification')
    if proof['status']!='PASS' or proof['source_files']!=source_files():
        raise ValueError('Changed or unqualified inference execution')
    return proof['sha256']


def load_package(family,package,protocol,arm,seed,qualification):
    if family=='base':
        if package or arm not in ('BASE','L0-S'):raise ValueError('BASE/control use only the immutable full base')
        return None,None
    if not package:raise ValueError('A pure deployment package is required')
    if family=='adapters':
        # A finite qualification subset may exercise a formal frozen package.
        # The inverse would promote qualification weights into formal evidence.
        package_qualification=json.loads((Path(package)/'deployment.json').read_text())['qualification']
        if package_qualification and not qualification:
            raise ValueError('Qualification weights cannot become formal deployment evidence')
        value,metadata=read_adapter_package(package,protocol,arm=arm,seed=seed,qualification=package_qualification)
        return value,metadata
    from .latent.deployment import load_q_package
    value,metadata=load_q_package(package,protocol,'cpu',allow_qualification=qualification)
    if metadata['arm']!=arm or metadata['training_seed']!=seed or (metadata['qualification'] and not qualification):
        raise ValueError('Latent deployment package belongs to another policy or qualification')
    return value,metadata


def identity(protocol,family,arm,seed,metadata,tokens,qualification,proof,root_scale=1.):
    learned=family=='latent'
    return dict(schema=SCHEMA,protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
        family=family,arm=arm,training_seed=seed,qualification=qualification,
        policy_qualification=metadata['qualification'] if metadata else False,
        root_policy_sha256=metadata['policy_sha256'] if learned else None,
        adapter_sha256=metadata['adapter_sha256'] if family=='adapters' else None,
        package_identity_sha256=metadata['sha256'] if metadata else None,
        checkpoint_state=(metadata['checkpoint_state'] if learned else
                          dict(phase='COMPLETE',arm=metadata['checkpoint_arm'],round=metadata['checkpoint_rounds'],updates=metadata['checkpoint_updates'])) if metadata else None,
        tokens_sha256=fingerprint(tokens),scenes=len(tokens),evaluation_seeds=list(protocol['evaluation_seeds']),
        scorer_config_hash=protocol['scorer']['sha256'],sampler_config_hash=fingerprint(protocol['samplers']),
        root_scale=root_scale,readout='q_sampled_single' if learned else 'single_candidate_source_shift5_ODE_K10',
        root_innovations=ROOT_DESCRIPTION,deployment_contract=contract(protocol),
        inference='one source shift5 ODE10 output per scene-bound innovation; no search/teacher/scorer selection',
        actor_dependencies='immutable full base plus pure learned-q or merged-adapter package',
        execution_equivalence_sha256=proof,source_files=source_files())


def make_runtime(protocol,family,value,device,ledger,experiment,split,workers):
    env=NativeEnvironment(protocol,device=device,trainable='frozen',scorer_workers=workers,
        ledger_path=ledger,experiment_id=experiment,evaluation_scope='NAVTEST_ONLY' if split=='NAVTEST' else None)
    if env.actor is not None:raise AssertionError('Deployment must not construct a reference actor')
    policy=None
    if family=='adapters':merge_native_adapters_(env.model.policy.action_expert,value,expected_layers=30)
    elif family=='latent':policy=value.to(env.device).eval().requires_grad_(False)
    if any(p.requires_grad for p in env.model.parameters()) or (policy is not None and any(p.requires_grad for p in policy.parameters())):
        raise AssertionError('Inference cannot update a model or its observation provider')
    return env,policy


@torch.no_grad()
def roots_for(policy,condition,token,seeds,identity):
    params=policy(condition_features(condition)).detached() if policy is not None else None
    rows=[]
    for seed in seeds:
        rng=torch.Generator(device='cpu').manual_seed(seed_for('native_qwen_sourceflow_ODE',seed,token))
        root=sample_roots(params,1,rng)[:,0] if params is not None else torch.randn((1,8,3),generator=rng,dtype=torch.float32)*identity['root_scale']
        metadata=dict(root=root[0].detach().cpu().tolist(),single_candidate=True,selection='none',root_scale=identity['root_scale'],
                      root_policy_sha256=identity['root_policy_sha256'])
        if params is not None:
            metadata.update(mean=params.mean[0].cpu().tolist(),log_std=params.log_std[0].cpu().tolist(),
                prior_KL_joint=float(joint_kl(params,params.standard_prior())[0]),entropy_joint=float(joint_entropy(params)[0]))
        rows.append((seed,root,metadata))
    return rows
