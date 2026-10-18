"""Lock upstream membership/order and the README command; no new model rollout."""
import ast
import copy
from collections import Counter
import json
from pathlib import Path
import pickle
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Dict, List
import yaml
from starVLA.rl.fm_transfer.source_process import ROOT, UPSTREAM, SOURCE_LIST, expected_config, MODES, B_SFT, OWN_MANIFEST
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint, UPSTREAM_COMMIT, RAW_CONDITION, CACHE_ROOTS
from starVLA.rl.flow_grpo.transactions import atomic_json


def source_function(path, name):
    tree=ast.parse(path.read_text()); node=copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name))
    ns=dict(Dict=Dict,List=List,Any=Any,Path=Path,SceneFilter=object,Scene=object,pickle=pickle,
        tqdm=lambda values,**kwargs:values)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
    return ns[name]


def main():
    require_dispatch(); started=time.monotonic()
    if (ROOT/'source_manifest.json').exists():raise FileExistsError('Registered membership already exists')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM_COMMIT:
        raise ValueError('Changed upstream checkout')
    paths=[SOURCE_LIST,'navsim/navsim/common/dataloader.py',
        'navsim/navsim/planning/script/config/training/default_train_val_test_log_split.yaml',
        'configs/train_grpo.yaml','configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml',
        'configs/data/navsim.yaml','README.md','src/simwam/trainer_grpo.py']
    subprocess.run(['git','-C',str(UPSTREAM),'diff','--exit-code','HEAD','--',*paths],check=True)
    source=yaml.safe_load((UPSTREAM/SOURCE_LIST).read_text()); source_tokens=set(source['tokens'])
    pop=json.loads(Path(POPULATION).read_text()); train=set(pop['train_tokens']); dev=set(pop['dev_tokens']); test=set(pop['navtest_tokens'])
    if len(source_tokens)!=17389 or source_tokens&test or source_tokens-(train|dev):raise ValueError('Unexpected source population')
    split=yaml.safe_load((UPSTREAM/paths[2]).read_text())
    source_logs=set(split['train_logs'])
    retained=source_tokens&train
    if any(pop['token_logs'][t] not in source_logs for t in retained):raise ValueError('Source train_logs excludes an expected token')
    raw=Path('/mnt/navsim/trainval_navsim_logs/trainval')
    selected_logs=[p for p in raw.iterdir() if p.name.removesuffix('.pkl') in source_logs]
    filt=SimpleNamespace(num_frames=source['num_history_frames']+source['num_future_frames'],
        num_history_frames=source['num_history_frames'],frame_interval=source['frame_interval'],
        has_route=source['has_route'],max_scenes=None,log_names=list(source_logs),tokens=list(retained))
    loader=UPSTREAM/'navsim/navsim/common/dataloader.py'
    frames=source_function(loader,'filter_scenes')(raw,filt)
    tokens=list(frames);del frames
    if set(tokens)!=retained:raise ValueError('Raw source SceneLoader has missing/extra tokens')
    # All assets must exist before registration; no silent skip/zero at training.
    from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
    store=ConditionStore(RAW_CONDITION)
    screen=Path('/mnt/project/fm-transfer-20261016/screens/B_SOURCE_ACTION_DIT_SFT_update100000')
    counts=Counter();scores=[];payload_hasher=__import__('hashlib').sha256()
    for token in tokens:
        store.get(token)
        log=pop['token_logs'][token]
        if not any((Path(r)/log/'unknown'/token/'metric_cache.pkl').is_file() for r in CACHE_ROOTS):
            raise FileNotFoundError('Missing canonical source TRAIN cache: '+token)
        record=screen/'scenes'/token[:2]/(token+'.json');row=json.loads(record.read_text())
        if row['token']!=token or row['payload_sha256']!=file_sha(record.with_suffix('.pt')):
            raise ValueError('Changed existing SFT screen payload')
        payload_hasher.update(file_sha(record).encode());m=row['metrics'];scores.append(m['score'])
        safe=all(m[k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))
        counts['safe_high' if safe and m['score']>=.9 else 'safe_low' if safe else 'unsafe']+=1
    obj=dict(status='COMPLETE_UPSTREAM_DATA_TRAIN_ONLY',upstream_commit=UPSTREAM_COMMIT,
        source_yaml_sha256=file_sha(UPSTREAM/SOURCE_LIST),source_loader_sha256=file_sha(loader),
        population_identity=pop['identity_sha256'],source_scenes=len(source_tokens),retained_scenes=len(tokens),
        excluded_DEV_scenes=len(source_tokens&dev),source_Navtest_overlap=0,missing_source_scenes=0,
        tokens=tokens,log_count=len({pop['token_logs'][t] for t in tokens}),
        input_log_enumeration=[p.name for p in selected_logs],
        source_order='unmodified source filter_scenes, local filesystem log enumeration; author machine enumeration unknown',
        B_SFT_cohorts=dict(counts),B_SFT_mean_score=sum(scores)/len(scores),
        existing_screen_digest=payload_hasher.hexdigest(),source_identity={p:file_sha(UPSTREAM/p) for p in paths})
    obj['sha256']=fingerprint(obj);atomic_json(ROOT/'source_manifest.json',obj)
    repo=Path.cwd();public=repo/'reports/ddp_simwam_source_process_20261018';public.mkdir(parents=True,exist_ok=True)
    public_obj={k:v for k,v in obj.items() if k not in ('tokens','input_log_enumeration')}
    public_obj.update(private_manifest_sha256=file_sha(ROOT/'source_manifest.json'),
        token_order_sha256=fingerprint(tokens),input_log_order_sha256=fingerprint(obj['input_log_enumeration']),
        preparation_seconds=time.monotonic()-started,new_model_inference=0,new_scores=0)
    atomic_json(public/'source_data_audit.json',public_obj)
    own=json.loads(OWN_MANIFEST.read_text())
    for mode in MODES:
        name='simwam_upstream_data_long.yaml' if mode=='UPSTREAM_DATA' else 'simwam_own_data_long.yaml'
        (repo/'configs/flow_rl'/name).write_text(yaml.safe_dump(expected_config(mode),sort_keys=False))
    register=dict(schema='simwam_source_process_registration_v1',registered_unix=time.time(),upstream_commit=UPSTREAM_COMMIT,
        user_instruction='尽可能完全照搬simwam的RL训练过程，用什么数据，怎么训，训多久都尽可能一致；用可用资源尽快训练',
        SFT=str(B_SFT),SFT_files_sha256=file_sha(B_SFT/'files.json'),
        formal_runs={mode:dict(config='configs/flow_rl/'+('simwam_upstream_data_long.yaml' if mode=='UPSTREAM_DATA' else 'simwam_own_data_long.yaml'),
            manifest=str(ROOT/'source_manifest.json' if mode=='UPSTREAM_DATA' else OWN_MANIFEST),
            manifest_sha256=file_sha(ROOT/'source_manifest.json' if mode=='UPSTREAM_DATA' else OWN_MANIFEST),
            max_updates=170000,training_seed=42,evaluation_seed=2101) for mode in MODES},
        released_weight_budget_note='README labels released RL step20000; default config20000; CLI170000. Do not claim 170k produced published91.9.',
        main_run='UPSTREAM_DATA',control='OWN_DATA_CONTROL isolates dataset choice under same longer budget',
        own_data_scenes=len(own['tokens']),data_overlap=len(set(own['tokens'])&set(tokens)),
        milestones=[0,5000,20000,50000,100000,170000],
        Navtest_fixed_observations=[20000,170000],Navtest_checkpoint_selection=False,
        evaluation_grids=['E_NATIVE','E_SHIFT5'],evaluation_metrics=['canonical_v1.1','official_v1.0_separate_rescore'],
        adaptations=['same B100k IL and original frozen Qwen; no Wan/video model substitution',
            'DDP eight-by-four normalization/heading pair and original decoder retained',
            'previously qualified FP32 actor/latent/optimizer and DDP layout retained',
            'source tokens excluding352 locked DEV scenes; local source-loader order frozen',
            'canonical independent scoring; infrastructure failures abort, never reward0'],
        algorithm_additions=[],old_runs_unchanged=True)
    register['sha256']=fingerprint(register);atomic_json(ROOT/'registration.json',register);atomic_json(public/'registration.json',register)
    print(json.dumps(public_obj))

if __name__=='__main__':main()
