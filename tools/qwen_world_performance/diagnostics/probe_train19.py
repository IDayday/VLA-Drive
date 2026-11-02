"""Finite fixed-IL70 TRAIN-only G16/G32 exploration; never trains/dispatches."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path('/mnt/project/qwen-world-performance-rl-20261103')
SOURCE = P/'runtime/formal_v1'
AUDIT = P/'analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT'
BASE = P/'diagnostics/IL70_TRAIN2048'
ROOT = P/'analysis/IL70_TRAIN19_G16_G32_EXPLORATION'
DATA = Path('/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN')
SCHEMA = 'world_IL70_TRAIN19_true_G16_G32_exploration_v1'
GROUPS = (16, 32)
HARD_CAP = 1024


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()


def fp(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seal(x):return dict(x,sha256=fp(x))


def read(path):
    x=json.loads(Path(path).read_text());body=dict(x)
    if body.pop('sha256',None)!=fp(body):raise ValueError('Changed seal: '+str(path))
    return x


def atomic(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def native(x):
    if hasattr(x,'detach'):return x.detach().cpu().tolist()
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    return x


def strict4(m):return all(float(m[k])==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort'))
def high(m):return strict4(m) and float(m['score'])>=.9
def feasible(m):return all(float(m[k])==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))


def source_inventory():
    paths=['starVLA/rl/qwen_world_postcot/scope_actor.py','starVLA/rl/qwen_world_postcot/rl_actor.py',
      'starVLA/rl/qwen_world_postcot/scope_conditions.py','starVLA/rl/qwen_world_postcot/loading.py',
      'starVLA/rl/qwen_world_safety_exploration/group_runtime.py','starVLA/rl/qwen_world_safety_exploration/conditions.py',
      'starVLA/rl/qwen_world_safety_exploration/credit.py','starVLA/rl/simwam_recipe/trainer_adapter.py',
      'starVLA/rl/simwam_recipe/vendor/action_math.py','starVLA/rl/simwam_recipe/vendor/scheduler_continuous.py',
      'starVLA/model/modules/qwen_sourceflow/action_codec.py','starVLA/rl/simwam_recipe/reward_readouts.py',
      'starVLA/rl/safe_flow_improvement/v1_reward.py','scripts/analysis/constrained_flow_noise/batched_canonical_score.py']
    return {n:sha(SOURCE/n) for n in paths}


def prepare(args):
    audit=read(AUDIT/'COMPLETE.json');gt=read(AUDIT/'PREPARED.json');base=read(BASE/'identity.json')
    if audit['status']!='COMPLETE' or audit['scored_count']!=2035:raise ValueError('Completed GT audit required')
    rows=gt['rows'];zero=[t for t in gt['tokens'] if rows[t]['baseline']['group']=='zero']
    controls=sorted([t for t in gt['tokens'] if rows[t]['baseline']['strict_high'] and rows[t]['eligible_for_scoring']],
        key=lambda t:fp(['TRAIN_GT_pilot',42,t]))[:4]
    tokens=zero+controls
    if len(zero)!=15 or len(controls)!=4 or len(set(tokens))!=19:raise ValueError('Fixed19 cohort changed')
    scenes={}
    for t in tokens:
        r=read(AUDIT/'scenes'/(t+'.json'))
        if r['status']!='SCORED' or not r['eligible_for_scoring']:raise ValueError('Legal sealed GT/B1 required')
        scenes[t]=dict(log=r['log'],role='B1_zero' if t in zero else 'fixed_strict_high_control',
            GT_scene_path=str(AUDIT/'scenes'/(t+'.json')),GT_scene_sha256=r['sha256'],
            B1_scene_path=r['baseline']['scene_path'],B1_scene_sha256=r['baseline']['scene_sha256'],
            B1_metrics=r['baseline']['metrics'],GT_metrics=r['GT_metrics'],cache=r['baseline']['cache'])
    dm=json.loads((DATA/'manifest.json').read_text())
    cfg=P/'configs/S0_LARGE_SAFETYEP.yaml'
    body=dict(schema=SCHEMA,status='PREPARED_NOT_DISPATCHED',source=str(SOURCE),
        source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        source_files_sha256=source_inventory(),script_sha256=sha(__file__),
        config=str(cfg),config_sha256=sha(cfg),parent_checkpoint=base['native_parent_checkpoint'],
        parent_native_model_sha256=base['parent_model_sha256'],baseline_export_model_sha256=base['model_sha256'],
        baseline_identity_sha256=base['sha256'],GT_audit_complete_file_sha256=sha(AUDIT/'COMPLETE.json'),
        data=str(DATA),data_manifest_sha256=sha(DATA/'manifest.json'),
        population_manifest=dm['source_population_path'],population_sha256=dm['population_sha256'],
        tokens=tokens,scenes=scenes,count=19,group_sizes=list(GROUPS),
        expected_new_PDM_queries=19*sum(GROUPS),hard_cap=HARD_CAP,scorer_workers_maximum=2,
        sampling=dict(mode='rigorous',steps=10,shift=5.,stochastic_steps=[7,8,9],eta=.1,
            randn_clip=5.,min_std=.001,root_seed=42,
            seed_namespace='TRAIN_IL70_true_group_exploration_v1',
            seed_key='namespace,42,token,G; independent G16 and G32 groups, not common random numbers',
            G32='one actual batch32 chain; never split into two G16 calls'),
        reused_B1=dict(candidates=1,steps=10,shift=5.,seed=2101,
            seed_namespace='native_qwen_sourceflow_ODE',new_queries=0),reused_GT_new_queries=0,
        training_updates=0,optimizer_created=False,NAVTEST_data_read=False,
        selection='all15 fixed TRAIN2048 B1zero plus first4 strict-high controls using the previously fixed GT-pilot hash order',
        normalization='No formal batch/optimizer. Save actual SafetyEP rawA/mask; do not invent world8 normalized advantages.',
        interpretation='G16/G32 use different BF16 batch shapes and independent query budgets; no causal fair-compute improvement claim.')
    m=seal(body);path=Path(args.manifest);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and read(path)!=m:raise FileExistsError('Immutable manifest differs')
    atomic(path,m)
    atomic(path.with_name('LAUNCH_PLAN.json'),seal(dict(schema=SCHEMA,status='PREPARED_NOT_DISPATCHED',
        manifest=str(path),manifest_sha256=m['sha256'],script=str(Path(__file__).resolve()),script_sha256=sha(__file__),
        cwd=str(SOURCE),command=['/mnt/project/flow-safe-improvement-20261013/venv/bin/python',str(Path(__file__).resolve()),
            'run','--manifest',str(path),'--output',str(ROOT/'results'),'--scorer-workers','2','--cpu-threads','2'],
        one_GPU=True,DDP=False,expected_new_PDM_queries=912,hard_cap=1024,automatic_dispatch=False)))
    print(json.dumps(dict(manifest=str(path),script_sha256=sha(__file__),count=19,queries=912,hard_cap=1024,tokens=tokens)),flush=True)


def group_stats(records,raw,b1):
    ms=[r['metrics'] for r in records];a=raw['A_raw'][0].detach().cpu().double().numpy()
    rewards=[float(m['score']) for m in ms];good=[high(m) for m in ms];safe=[feasible(m) for m in ms]
    strict=[strict4(m) for m in ms]
    teacher=[ok and m['score']-b1['score']>=.02 for ok,m in zip(good,ms)]
    return native(dict(G=len(ms),all_canonical_zero=all(x==0 for x in rewards),
        canonical_tie=max(rewards)==min(rewards),PG0=bool((a==0).all()),
        raw_A_positive=int((a>0).sum()),raw_A_negative=int((a<0).sum()),
        feasible_candidates=sum(safe),strict4_candidates=sum(strict),strict_high_candidates=sum(good),
        strict_high_teacher_gain_ge02=sum(teacher),
        strict_high_positive_rawA=sum(ok and x>0 for ok,x in zip(good,a)),
        strict_high_negative_rawA=sum(ok and x<0 for ok,x in zip(good,a)),
        unsafe_candidates=sum(not x for x in safe),unsafe_fraction=sum(not x for x in safe)/len(ms),
        unsafe_positive_rawA=sum((not s) and x>0 for s,x in zip(safe,a)),
        comfort_failed=sum(m['comfort']!=1 for m in ms),canonical_zero_candidates=sum(x==0 for x in rewards),
        best_score=max(rewards),best_feasible_score=max([m['score'] for m,s in zip(ms,safe) if s],default=None),
        best_strict4_score=max([m['score'] for m,s in zip(ms,strict) if s],default=None),
        best_teacher_candidate_index=max([i for i,t in enumerate(teacher) if t],key=lambda i:rewards[i],default=None)))


def run(args):
    m=read(args.manifest);out=Path(args.output)
    if m['schema']!=SCHEMA or Path.cwd().resolve()!=SOURCE.resolve():raise ValueError('Use exact frozen formal_v1 cwd')
    if m['script_sha256']!=sha(__file__) or m['source_files_sha256']!=source_inventory():raise ValueError('Source changed')
    if not 1<=args.scorer_workers<=2 or m['expected_new_PDM_queries']!=912 or m['hard_cap']!=1024:raise ValueError('Finite budget/workers changed')
    if sha(DATA/'manifest.json')!=m['data_manifest_sha256'] or sha(m['config'])!=m['config_sha256']:raise ValueError('Data/config changed')
    sys.path.insert(0,str(SOURCE))
    import numpy as np
    import torch
    import yaml
    from starVLA.rl.qwen_world_postcot.loading import load_deployment_checkpoint
    from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
    from starVLA.rl.qwen_world_research_data import ResearchSceneData
    from starVLA.rl.qwen_world_safety_exploration.conditions import GroupMemoryConditions
    from starVLA.rl.qwen_world_safety_exploration.group_runtime import collect_group
    from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit,canonical_components
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    from starVLA.rl.simwam_recipe.contracts import seed_for
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.qwen_world_performance.contracts import validate_recipe
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    out.mkdir(parents=True,exist_ok=True);(out/'groups').mkdir(exist_ok=True);(out/'arrays').mkdir(exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'BUDGET_LEDGER.json').exists():raise RuntimeError('Existing budget retained; no automatic repeat or ambiguous retry')
    cfg=yaml.safe_load(Path(m['config']).read_text());validate_recipe(cfg)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    configure_numerics();torch.set_num_threads(args.cpu_threads);torch.cuda.set_device(0)
    data=ResearchSceneData(DATA,role='NAVTRAIN')
    if sha(m['population_manifest'])!=m['population_sha256']:raise ValueError('Population changed')
    pop=json.loads(Path(m['population_manifest']).read_text())
    import re
    physical=lambda x:re.sub(r'_\d+_\d+$','',x)
    forbidden={physical(pop['token_logs'][t]) for t in pop['navtest_tokens']}
    for t in m['tokens']:
        if t not in data.records or t in set(pop['navtest_tokens']) or physical(data.logs[t]) in forbidden:raise ValueError('Non-TRAIN scene')
        old=read(m['scenes'][t]['GT_scene_path']);b=read(m['scenes'][t]['B1_scene_path'])
        if old['sha256']!=m['scenes'][t]['GT_scene_sha256'] or b['sha256']!=m['scenes'][t]['B1_scene_sha256'] or data.logs[t]!=m['scenes'][t]['log']:raise ValueError('Sealed B1/GT join changed')
    started=time.monotonic();model,parent=load_deployment_checkpoint(m['parent_checkpoint'],device='cuda')
    if parent['model_sha256']!=m['parent_native_model_sha256'] or parent['seal']['updates']!=70000:raise ValueError('Complete IL70 required')
    actor=ScopeRecipeActor(model.actor,cfg,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
    if actor.closure['trainable_numel']!=52707840 or actor._stoch_step_indices(10)!=[7,8,9]:raise ValueError('Source scope/W3 changed')
    actor.eval().requires_grad_(False);model.eval().requires_grad_(False);actor.record_sampling_terms=True
    providers={g:GroupMemoryConditions(model,data,parent['model_sha256'],actor,group_size=g) for g in GROUPS}
    service=RewardService(SOURCE,workers=args.scorer_workers,cache_roots=data.manifest['canonical_metric_cache_roots'])
    base=read(BASE/'identity.json')
    if service.protocol['sha256']!=base['protocol']['sha256']:raise ValueError('Original canonical protocol changed')
    identity=seal(dict(schema=SCHEMA,manifest_sha256=m['sha256'],parent_model_sha256=parent['model_sha256'],
        parent_native_seal=parent['seal'],scope=actor.closure,source=str(SOURCE),source_commit=m['source_commit'],
        source_files_sha256=source_inventory(),script_sha256=sha(__file__),protocol=service.protocol,
        group_sizes=list(GROUPS),Adam_updates=0,optimizer_created=False,current_parameters_during_probe='frozen/no_grad; no teacher or RL parameter restore',
        device=torch.cuda.get_device_name(0),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        model_loading_seconds=time.monotonic()-started,nominal_B1_reused=True,GT_reused=True))
    atomic(out/'identity.json',identity)
    ledger=dict(schema=SCHEMA,maximum=1024,expected=912,reserved_queries=0,completed_queries=0,groups={})
    atomic(out/'BUDGET_LEDGER.json',ledger)
    try:
      with torch.no_grad():
       for token in m['tokens']:
        for g in GROUPS:
          key=token+'_G'+str(g);reserved=ledger['reserved_queries']+g
          if reserved>912 or reserved>HARD_CAP:raise RuntimeError('Query cap exceeded')
          actor.grpo_group_size=g
          conds=providers[g];other=providers[32 if g==16 else 16]
          if other.token==token:
              conds.token=other.token;conds.cached=other.cached
          object.__setattr__(actor,'conditions',conds)
          seed=seed_for('TRAIN_IL70_true_group_exploration_v1',42,token,g)
          gen=torch.Generator(device='cpu').manual_seed(seed);initial=gen.get_state().clone()
          ledger['reserved_queries']=reserved;ledger['groups'][key]=dict(G=g,status='RESERVED',seed=seed)
          atomic(out/'BUDGET_LEDGER.json',ledger)
          version=dict(schema=SCHEMA,IL_updates=70000,Adam_updates=0,parent_model_sha256=parent['model_sha256'])
          leaf=collect_group(actor,conds.get(token,actor.device),token,m['scenes'][token]['log'],service,gen,
              group_size=g,variant='R_CANONICAL',version=version)
          if leaf['qualification_sampling_check']['status']!='PASS':raise ValueError('Actual sampled vs recomputed W3 terms differ')
          fresh=actor.action_chain_logprobs(leaf['cond_bg'],leaf['chain'],leaf['timesteps'],leaf['deltas'],stoch_idx=leaf['stoch_idx'])
          fresh_error=float((torch.exp(fresh-leaf['old_logp'])-1).abs().max())
          if fresh_error>1e-4:raise ValueError('Fresh source ratio check failed')
          mat=canonical_components(leaf['records'],g,device=actor.device);parts=GroupCredit(g).parts(mat['components'])
          stats=group_stats(leaf['records'],parts,m['scenes'][token]['B1_metrics'])
          requests=ActionCodec.denorm_odo(leaf['chain'][:,-1]).float().cpu().numpy()
          for i,r in enumerate(leaf['records']):
              if r['cache']['sha256']!=m['scenes'][token]['cache']['sha256'] or not np.array_equal(requests[i],r['requested_trajectory']):raise ValueError('Actual requested action/cache changed')
          ap=out/'arrays'/(key+'.npz')
          np.savez_compressed(ap,chain=leaf['chain'].cpu().numpy(),requested_trajectory=requests,
              old_logp=leaf['old_logp'].cpu().numpy(),fresh_logp=fresh.cpu().numpy(),
              timesteps=leaf['timesteps'].cpu().numpy(),deltas=leaf['deltas'].cpu().numpy(),stoch_idx=leaf['stoch_idx'].cpu().numpy(),
              old_mean=leaf['old_probability']['mean'].cpu().numpy(),old_std=leaf['old_probability']['std'].cpu().numpy(),
              components=mat['components'].cpu().numpy(),**{k:parts[k].cpu().numpy() for k in ('A_safe','A_EP','A_raw','feasible','cost')},
              RNG_before=initial.numpy(),RNG_after=gen.get_state().numpy())
          row=seal(native(dict(schema=SCHEMA,identity_sha256=identity['sha256'],token=token,log=m['scenes'][token]['log'],
              role=m['scenes'][token]['role'],G=g,one_actual_batch_call=True,chain_shape=list(leaf['chain'].shape),
              probability_shape=list(leaf['old_logp'].shape),seed=seed,statistics=stats,
              B1_metrics=m['scenes'][token]['B1_metrics'],GT_metrics=m['scenes'][token]['GT_metrics'],
              noise_audit=leaf['noise_audit'],sampling_check=leaf['qualification_sampling_check'],fresh_ratio_error=fresh_error,
              source_clamp_hit_rate=leaf['old_probability']['source_clamp_hit_rate'],
              source_probability='W3 clamped dimension-mean proxy; not endpoint joint density',
              records=leaf['records'],seconds=leaf['seconds'],arrays=dict(path=str(ap),sha256=sha(ap)),
              normalization_applied=False,PG0_definition='native GPU SafetyEP raw A exactly zero for all G candidates')))
          atomic(out/'groups'/(key+'.json'),row)
          ledger['completed_queries']+=g;ledger['groups'][key].update(status='COMPLETE',output_sha256=row['sha256'])
          atomic(out/'BUDGET_LEDGER.json',ledger)
          print(json.dumps(dict(token=token,G=g,queries=ledger['completed_queries'],statistics=stats)),flush=True)
          del leaf,fresh,parts,mat
    except Exception as exc:
        atomic(out/'FAILED.json',seal(dict(schema=SCHEMA,error=type(exc).__name__,message=str(exc),ledger=ledger)))
        raise
    finally:service.close()
    summarize(argparse.Namespace(manifest=args.manifest,output=str(out)))


def summarize(args):
    m=read(args.manifest);out=Path(args.output);rows={};identity=read(out/'identity.json')
    if identity['manifest_sha256']!=m['sha256']:raise ValueError('Wrong diagnostic output/manifest')
    for t in m['tokens']:
        rows[t]={}
        for g in GROUPS:
            p=out/'groups'/(t+'_G'+str(g)+'.json')
            if not p.exists():continue
            r=read(p)
            if r['G']!=g or r['token']!=t or r['identity_sha256']!=identity['sha256']:raise ValueError('Group mapping/identity changed')
            rows[t][str(g)]=r['statistics']
    complete=all(len(v)==2 for v in rows.values())
    cases=[]
    for t,groups in rows.items():
        scene=m['scenes'][t];a=groups.get('16');b=groups.get('32')
        found16=bool(a and a['strict_high_candidates']);found32=bool(b and b['strict_high_candidates'])
        cases.append(dict(token=t,log=scene['log'],role=scene['role'],B1=scene['B1_metrics'],GT=scene['GT_metrics'],
            G16=a,G32=b,GT_strict_high=high(scene['GT_metrics']),G16_found_strict_high=found16,G32_found_strict_high=found32,
            both_groups_no_high_but_GT_high=bool(a and b and not found16 and not found32 and high(scene['GT_metrics'])),
            B1_bad_but_sampled_high_witness=not high(scene['B1_metrics']) and (found16 or found32)))
    aggregate={}
    for name in ('B1_zero','fixed_strict_high_control'):
      groupcases=[c for c in cases if c['role']==name]
      aggregate[name]=dict(scenes=len(groupcases),
          G16_found=sum(c['G16_found_strict_high'] for c in groupcases),G32_found=sum(c['G32_found_strict_high'] for c in groupcases),
          both_no_high_GT_high=sum(c['both_groups_no_high_but_GT_high'] for c in groupcases),
          current_B1_bad_with_sampled_high=sum(c['B1_bad_but_sampled_high_witness'] for c in groupcases),
          group_statistics={str(g):dict(complete_groups=sum(c['G'+str(g)] is not None for c in groupcases),
              all_canonical_zero_groups=sum(c['G'+str(g)]['all_canonical_zero'] for c in groupcases if c['G'+str(g)]),
              PG0_groups=sum(c['G'+str(g)]['PG0'] for c in groupcases if c['G'+str(g)]),
              unsafe_candidates=sum(c['G'+str(g)]['unsafe_candidates'] for c in groupcases if c['G'+str(g)]),
              strict_high_candidates=sum(c['G'+str(g)]['strict_high_candidates'] for c in groupcases if c['G'+str(g)])) for g in GROUPS})
    ledger=json.loads((out/'BUDGET_LEDGER.json').read_text())
    if complete and ledger['completed_queries']!=912:raise ValueError('Complete output must account for all912 actual candidates')
    value=seal(dict(schema=SCHEMA,status='COMPLETE' if complete else 'PARTIAL',manifest_sha256=m['sha256'],
        queries=ledger,cohorts=aggregate,cases=cases,training_updates=0,
        limitations=['Fixed19 targeted TRAIN diagnostics, not fullpopulation or NAVTEST.',
          'Independent G16/G32 BF16 batch shapes/seeds/query costs; not matched-compute causal evidence.',
          'Sampled high alongside bad B1 is a pre-update opportunity, not proof of persistent training absorption failure.',
          'No formal world8 advantage normalization; PG0 uses exact native raw-credit direction.',
          'Neither GT nor sampled teachers are admitted into S6/current algorithms.']))
    atomic(out/'SUMMARY.json',value)
    if complete:atomic(out/'COMPLETE.json',seal(dict(schema=SCHEMA,status='COMPLETE',summary_sha256=value['sha256'],new_PDM_queries=912,hard_cap=1024,scene_count=19,group_count=38)))
    print(json.dumps(dict(status=value['status'],cohorts=aggregate)),flush=True)


def selfcheck():
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(SOURCE))
    import torch
    from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit
    for g in GROUPS:
        x=torch.ones(1,g,4,dtype=torch.float64);x[:,:,1]=0;x[0,0,0]=0
        parts=GroupCredit(g).parts(x)
        assert not bool(parts['A_raw'].eq(0).all())  # all canonicalR0, genuine cost variation
        records=[dict(metrics=dict(no_at_fault_collisions=float(x[0,i,0]),drivable_area_compliance=0.,
            time_to_collision_within_bound=1.,comfort=1.,ego_progress=1.,score=0.)) for i in range(g)]
        stats=group_stats(records,parts,dict(score=0.))
        assert stats['all_canonical_zero'] and not stats['PG0'] and stats['G']==g
        json.dumps(dict(statistics=stats),allow_nan=False) # actual native numpy-derived readout path
        y=torch.ones(1,g,4,dtype=torch.float64)
        assert bool(GroupCredit(g).parts(y)['A_raw'].eq(0).all()) # successful EP1 saturation
    assert 19*sum(GROUPS)==912<=1024
    print(json.dumps(dict(status='PASS',check='G16/G32 exact raw-credit zero-vs-PG0 and budget; CPU only')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for name in ('prepare','run','summarize'):
        p=sub.add_parser(name);p.add_argument('--manifest',default=str(ROOT/'MANIFEST.json'))
        if name!='prepare':p.add_argument('--output',default=str(ROOT/'results'))
        if name=='run':p.add_argument('--scorer-workers',type=int,default=2);p.add_argument('--cpu-threads',type=int,default=2)
    sub.add_parser('selfcheck');args=parser.parse_args()
    {'prepare':prepare,'run':run,'summarize':summarize,'selfcheck':lambda _:selfcheck()}[args.mode](args)
