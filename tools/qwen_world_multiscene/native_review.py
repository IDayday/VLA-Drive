"""CPU validation/comparison and strict export for the new packed recipe."""
import argparse,gc,json,re
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.qwen_world_postcot.contracts import file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_multiscene.recovery import validate_boundary,validate_ranks


def load_validated(path):
    root=Path(path).resolve();seal=json.loads((root/'COMPLETE').read_text());files=json.loads((root/'files.json').read_text())
    identity=json.loads((root/'identity.json').read_text());body=dict(identity);saved=body.pop('sha256',None)
    required={'identity.json','optimizer.pt','transaction.json'}|{f'rank{i}.pt'for i in range(8)}
    if ('.incomplete' in root.name or seal.get('schema')!=2 or seal.get('world')!=8
            or identity.get('world')!=8 or identity.get('schema')!='world_postcot_cleanx_scope_RL_v1'
            or saved!=fingerprint(body)or seal['files_sha256']!=file_sha(root/'files.json')or not required<=files.keys()):
        raise ValueError('Incomplete, changed or foreign packed checkpoint')
    for name,sha in files.items():
        file=root/name
        if not file.resolve().is_relative_to(root)or file_sha(file)!=sha:raise ValueError('Changed checkpoint payload: '+name)
    common=torch.load(root/'optimizer.pt',map_location='cpu',weights_only=False)
    ranks=[];summaries=[];transaction=json.loads((root/'transaction.json').read_text())
    for rank in range(8):
        own=torch.load(root/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:raise ValueError('Rank identity changed')
        summary=validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'],expected_rank=rank)
        if summary!=transaction['ranks'][rank]or summary['phase']!=seal['phase']or summary['updates']!=seal['updates']:
            raise ValueError('CPU reconstructed native boundary differs from sealed transaction')
        if set(own['rng'])!={'python','numpy','torch','cuda','rollout_generator'}:raise ValueError('Incomplete native RNG')
        ranks.append(own);summaries.append(summary)
    validate_ranks(summaries)
    p=identity['recipe']['research']['performance']
    if p['group_size']!=16 or p['scenes_per_collection']!=128 or p['inner_updates']!=4:
        raise ValueError('This bounded qualification/export adapter requires actual B128/G16/M4')
    if set(common['adapters'])!=set(identity['parameter_contract']['trainable_names']):raise ValueError('Incomplete actual trainable closure')
    if any(x.dtype!=torch.float32 or not torch.isfinite(x).all()for x in common['adapters'].values()):raise ValueError('Invalid trained parameters')
    pending=ranks[0]['rollout'] is not None
    counts=dict(pending=pending,updates=seal['updates'],next_inner=seal['next_inner'],
        raw_scenes=ranks[0]['state']['raw_scene_exposures'],valid_scenes=ranks[0]['state']['valid_scene_exposures'],
        padding_scenes=ranks[0]['state']['padded_scene_exposures'],physical_PG_candidates=ranks[0]['state']['new_candidates'])
    if pending:
        leaves=[s for own in ranks for s in own['rollout']['samples']];valid=[s for s in leaves if s['valid_scene']]
        for own in ranks:
            w=own['rollout'];stat=w['advantage_statistics']
            if (w['valid_global_scenes']!=len(valid)or stat['normalization_population_groups']!=len(valid)
                    or stat['normalization_population_candidates']!=len(valid)*16):raise ValueError('Global fresh population count changed')
        if len(leaves)!=128:raise ValueError('Not a real global B128 window')
        if any(s['adv_mask'].any()for s in leaves if not s['valid_scene']):raise ValueError('Padding has PG credit')
        counts.update(pending_actual_scenes=len(leaves),pending_valid_scenes=len(valid),pending_padding_scenes=128-len(valid),
            normalization_candidates=len(valid)*16,normalization_std=summaries[0]['normalization_std'],
            padding_scope='actual stored slots; zero here means the first GPU window did not exercise epoch-tail padding')
    return dict(path=str(root),seal=seal,identity=identity,common=common,ranks=ranks,summaries=summaries,counts=counts)


def assert_equal(a,b,path='root'):
    if torch.is_tensor(a):
        if not torch.is_tensor(b)or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise ValueError('Tensor differs: '+path)
    elif isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray)or a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise ValueError('Numpy state differs: '+path)
    elif isinstance(a,dict):
        if not isinstance(b,dict)or a.keys()!=b.keys():raise ValueError('Fields differ: '+path)
        for key in a:assert_equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        if type(a)is not type(b)or len(a)!=len(b):raise ValueError('Sequence differs: '+path)
        for i,(x,y)in enumerate(zip(a,b)):assert_equal(x,y,path+f'[{i}]')
    elif a!=b:raise ValueError('Value differs: '+path)


def learning_state(state):
    #Timings are explicitly administrative; every other saved field is checked.
    return {k:v for k,v in state.items()if not k.startswith('seconds_')}


def scientific_rollout(value):
    #Official time-to-collision/infraction seconds are scientific evidence,
    #not elapsed runtimes. Exclude only these known timing field names.
    timings={'seconds','scoring_seconds','auxiliary_seconds','history_seconds','auxiliary_seconds_local'}
    if isinstance(value,dict):return{k:scientific_rollout(v)for k,v in value.items()if k not in timings}
    if isinstance(value,list):return[scientific_rollout(v)for v in value]
    if isinstance(value,tuple):return tuple(scientific_rollout(v)for v in value)
    return value


def compare(left,right):
    a=load_validated(left);b=load_validated(right)
    assert_equal(a['identity'],b['identity'],'identity')
    for field in ('adapters','optimizer','scheduler'):assert_equal(a['common'][field],b['common'][field],field)
    administrative_invocations=[]
    for rank,(x,y)in enumerate(zip(a['ranks'],b['ranks'])):
        assert_equal(x['rng'],y['rng'],f'rank{rank}.all_native_RNG')
        assert_equal(learning_state(x['state']),learning_state(y['state']),f'rank{rank}.learning_state')
        if (x['rollout']is None)!=(y['rollout']is None):raise ValueError('Different pending state')
        if x['rollout']is not None:
            #Native rollout includes measured wallclock times. Compare all
            #scientific fields through independently validated summaries/hashes
            #after only dropping explicit timing dictionaries.
            scientific=[scientific_rollout(z['rollout'])for z in (x,y)]
            ids=[z.get('invocation')for z in scientific]
            if any(z is not None for z in ids):
                if any(not isinstance(z,str)or re.fullmatch('[0-9a-f]{32}',z)is None for z in ids):
                    raise ValueError('Missing/invalid top-level administrative Invocation UUID')
                administrative_invocations.append(dict(rank=rank,left=ids[0],right=ids[1]))
                for z in scientific:z.pop('invocation')
            #Only this exact top-level field is administrative. In particular,
            #teacher ancestry/version and nested invocation metadata stay exact.
            assert_equal(scientific[0],scientific[1],f'rank{rank}.pending_rollout')
    result=dict(schema='world_coherent_packed_native_comparison_v1',status='PASS',left=a['path'],right=b['path'],
        identity_sha256=a['identity']['sha256'],exact=['all actual trainables','Adam moments/steps/groups','scheduler',
            '8rank Python/Numpy/TorchCPU/CUDA/source-generator RNG','all learning state including scene_stream/selection plan','pending scientific source chains/probabilities/advantages'],
        excluded='explicit measured wallclock seconds and top-level independent-run Invocation UUID only',
        administrative_invocations=administrative_invocations,counts=a['counts'],world=8,
        phase=a['seal']['phase'],micro_scenes=a['identity']['packed_execution']['micro_scenes'])
    return result


def export_native(parent,output,checkpoint,*,allow_qualification=False):
    data=load_validated(checkpoint);identity=data['identity'];seal=data['seal']
    if seal['next_inner']or seal['phase']not in ('READY','COMPLETE'):raise ValueError('Export only at completed window boundary')
    if identity.get('purpose')!='formal'and not allow_qualification:raise ValueError('Qualification export must be explicitly diagnostic')
    if Path(parent).resolve()!=Path(identity['IL_parent']['path']).resolve():raise ValueError('Different pinned IL70 parent')
    validation=dict(schema='world_packed_export_native_validation_v1',status='PASS',checkpoint=data['path'],
        files_sha256=seal['files_sha256'],identity_sha256=identity['sha256'],counts=data['counts'],
        micro_scenes=identity['packed_execution']['micro_scenes'],deployment='unchanged single genuine B1 ODE K10shift5seed2101')
    del data;gc.collect()
    from starVLA.rl.qwen_world_postcot.deployment import export_checkpoint,inspect_export
    meta=export_checkpoint(parent,output,checkpoint);actual=inspect_export(output)
    if actual['RL_parent']!=identity or actual['RL_updates']!=seal['updates']or actual['RL_checkpoint_files_sha256']!=seal['files_sha256']:
        raise ValueError('Export lineage changed')
    destination=Path(output).with_name(Path(output).name+'.native_validation.json');destination.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(destination,validation)
    return dict(status='PASS',path=str(output),validation=str(destination),qualification=meta['qualification'])


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('compare');c.add_argument('--left',required=True);c.add_argument('--right',required=True);c.add_argument('--output',required=True)
    e=sub.add_parser('export');e.add_argument('--parent-checkpoint',required=True);e.add_argument('--rl-checkpoint',required=True);e.add_argument('--output',required=True);e.add_argument('--allow-qualification',action='store_true')
    a=p.parse_args()
    if a.action=='compare':
        value=compare(a.left,a.right);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,value);print(json.dumps(value))
    else:print(json.dumps(export_native(a.parent_checkpoint,a.output,a.rl_checkpoint,allow_qualification=a.allow_qualification)))


if __name__=='__main__':main()
