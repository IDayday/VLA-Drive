"""CPU synthetic complete16-rank benchmark, missing/fresh-error and resume."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.qwen_world_compare.recovery_state import tensor_digest
from tools.d46_priority_rl import review_execution_benchmark as b


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return value


def sealed(path,value):
    value=copy.deepcopy(value);value.pop('sha256',None);value['sha256']=fingerprint(value);return put(path,value)


def make_native(root,identity,u,*,pending=False,common=b'Adam at1500',journal_origin=1500):
    cp=root/'checkpoints'/f'update{u:06}_inner{int(pending)}';cp.mkdir(parents=True)
    put(cp/'identity.json',identity);(cp/'optimizer.pt').write_bytes(common)
    c=(u+1)//2;lanes=[torch.Generator().manual_seed(42+i).get_state()for i in range(8)]
    virtual=fingerprint([hashlib.sha256(x.numpy().tobytes()).hexdigest()for x in lanes]);rows=[];payloads=[]
    for rank in range(16):
        state=dict(updates=u,next_inner=int(pending),phase='INNER_COMMITTED'if pending else'READY',collections=c,scene_cursor=64*c,
            valid_scene_exposures=64*c,new_candidates=1024*c,extra_B1_queries=0,adaptive_G32_queries=0,journal_origin_updates=journal_origin,
            world16_virtual_sampler=dict(lanes=lanes,committed_collections=c))
        own=dict(identity_sha256=identity['sha256'],state=state,rollout={'pending':True}if pending else None,
            rng={'torch':torch.Generator().manual_seed(100+rank).get_state(),'rollout_generator':torch.Generator().manual_seed(200+rank).get_state()})
        torch.save(own,cp/f'rank{rank}.pt');payloads.append(own)
        rows.append(dict(run=identity['sha256'],phase=state['phase'],updates=u,next_inner=int(pending),scene_cursor=64*c,virtual8_rng_sha256=virtual))
    put(cp/'transaction.json',dict(schema=b.CONTRACT,ranks=rows))
    files={p.name:b.sha(p)for p in cp.iterdir()};put(cp/'files.json',files)
    put(cp/'COMPLETE',dict(schema=2,contract=b.CONTRACT,world=16,phase='INNER_COMMITTED'if pending else'READY',updates=u,next_inner=int(pending),files_sha256=b.sha(cp/'files.json')))
    return cp,payloads,virtual


@pytest.fixture
def complete(tmp_path,monkeypatch):
    source=tmp_path/'SCI';source.mkdir();(source/'fixture.py').write_text('frozen execution\n')
    sources={'fixture.py':b.sha(source/'fixture.py')};monkeypatch.setattr(b,'git_head',lambda p:'frozen_commit')
    cfg=dict(arm='D3_FPO',global_scenes=64,group=16,inner_updates=2,mc_samples=4)
    formal_id=sealed(tmp_path/'formal/identity.json',dict(schema='d44_native_rl_v1',world=16,qualification=False,fresh_Adam=False,
        recipe=cfg,scope={'same_LoRA':True},parent_model_sha256=b.MODEL_SHA,source_files=sources,native_world_transition={'origin_updates':1440,'target_qualification':False}))
    cp,source_owns,virtual=make_native(tmp_path/'formal',formal_id,1500,journal_origin=1440)
    roots=[]
    for label,cycle,enabled in [('baseline',30.,False),('candidate',20.,True)]:
        root=tmp_path/label;root.mkdir();roots.append(root)
        flags=dict.fromkeys(b.FLAGS,False);flags.update(fast_prefix=enabled,fuse_fm_mc=enabled,rotary_cache=enabled)
        meta=dict(schema='d46_world16_execution_upgrade_v1',origin_updates=1500,source_qualification=False,target_qualification=True,
            flags=flags,source_world16_checkpoint=str(cp),source_identity_sha256=formal_id['sha256'],source_checkpoint_identity_file_sha256=b.sha(cp/'identity.json'),
            source_checkpoint_files_sha256=b.sha(cp/'files.json'),common_optimizer_sha256=b.sha(cp/'optimizer.pt'),new_inventory_sha256=fingerprint(sources))
        identity=copy.deepcopy(formal_id);identity.pop('sha256');identity.update(qualification=True,execution_upgrade=meta);identity['native_world_transition']['target_qualification']=True
        identity=sealed(root/'identity.json',identity)
        upgrade=sealed(root/'EXECUTION_UPGRADE.json',dict(execution_upgrade=meta,target_identity_sha256=identity['sha256'],
            source_rank_RNG_sha256=[tensor_digest(x['rng'])for x in source_owns],source_virtual8_rng_sha256=virtual))
        sealed(root/'WORLD_TRANSITION.json',dict(target_identity_sha256=identity['sha256'],execution_upgrade_sha256=upgrade['sha256'],source_inventory=sources))
        make_native(root,identity,1500);make_native(root,identity,1512,common=b'updated Adam at1512')
        receipts=[]
        for node in range(2):
            receipts.append(dict(source=str(source),source_sha='frozen_commit',host=f'host{node}',expected_gpu_uuids=[f'UUID{node}_{g}'for g in range(8)],
                command=['python','-m','torch.distributed.run','--nnodes=2','--nproc_per_node=8',f'--node_rank={node}','-m','tools.d46_world16.train_fast',
                    '--output',str(root),'--resume',str(root/'checkpoints/update001500_inner0'),'--qualification','--stop-after-update','1512']))
        put(root/'BENCHMARK_PLAN.json',dict(receipts=receipts))
        for rank in range(16):
            rows=[]
            for u in range(1501,1513):
                c=(u+1)//2;inner=(u-1)%2+1
                rows.append(dict(updates=u,run=identity['sha256'],collection=c,inner_completed=inner,behavior_update=2*(c-1),invocation='first',
                    active_batch=dict(scenes=64,group=16),scene_exposures=64*c,PG_candidates=1024*c,generated_PG_candidates_including_padding=1024*c,
                    extra_B1_queries=0,adaptive_G32_queries=0,update_seconds=5. if enabled else 8.,collection_seconds_generation=8. if enabled else 10.,
                    collection_seconds_scoring=1.,peak_GPU_bytes=2**30,at=1000+(c-750)*cycle+(0 if inner==2 else-5)+rank*.001,
                    gradient_norm=.1,losses=dict(fresh_ratio_error=0.,fpo_joint_logratio_min=0.,fpo_joint_logratio_max=0.)))
            (root/f'metrics_rank{rank}.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in rows))
            put(root/'invocations/first'/f'rank{rank}.json',dict(run=identity['sha256'],rank=rank,start_update=1500,resume=str(root/'checkpoints/update001500_inner0'),physical_candidates=384,scorer_candidate_attempts=384,generation_worker_seconds=50.,scoring_wait_worker_seconds=5.))
            put(root/f'result_rank{rank}.json',dict(status='QUALIFICATION_STOPPED',updates=1512,next_inner=0))
    return roots


def test_same_native_complete_comparison_has_known_full_cycle_speedup(complete):
    value=b.review(*complete)
    assert value['status']=='COMPLETE'and value['full_cycle_speedup']==pytest.approx(1.5)
    assert value['matched_full_cycle_speedup']==pytest.approx(1.5)
    assert value['matched_cycle_end_updates']==[1506,1508,1510,1512]
    assert value['baseline']['steady_collections']==value['candidate']['steady_collections']==4
    assert value['candidate']['actual_queries']['PG_candidates']==6144
    assert value['candidate']['actual_queries']['scorer_candidate_attempts']==6144
    assert len(value['candidate']['journals'])==16 and len(value['candidate']['journals'][0]['rows'])==12


def test_missing_committed_row_and_nonfinite_or_fresh_FPO_error_fail_closed(complete):
    base,candidate=complete;path=candidate/'metrics_rank9.jsonl';original=path.read_text();rows=original.splitlines()
    path.write_text('\n'.join(rows[:-1])+'\n')
    with pytest.raises(ValueError,match='Missing/duplicate'):b.review(base,candidate)
    for key,value in [('gradient_norm',float('nan')),('fresh',.1)]:
        changed=[json.loads(x)for x in rows]
        if key=='fresh':changed[0]['losses']['fpo_joint_logratio_max']=value
        else:changed[0][key]=value
        path.write_text(''.join(json.dumps(x)+'\n'for x in changed))
        with pytest.raises(ValueError,match='Nonfinite|Fresh FPO'):b.review(base,candidate)
    path.write_text(original)


def test_real_pending_resume_excludes_cross_invocation_interval(complete):
    base,candidate=complete;identity=json.loads((candidate/'identity.json').read_text())
    make_native(candidate,identity,1511,pending=True,common=b'Adam at1511')
    for rank in range(16):
        path=candidate/f'metrics_rank{rank}.jsonl';rows=[json.loads(x)for x in path.read_text().splitlines()];rows[-1]['invocation']='resumed';rows[-1]['at']+=100.
        path.write_text(''.join(json.dumps(x)+'\n'for x in rows))
        put(candidate/'invocations/resumed'/f'rank{rank}.json',dict(run=identity['sha256'],rank=rank,start_update=1511,resume=str(candidate/'checkpoints/update001511_inner1'),physical_candidates=0,scorer_candidate_attempts=0,generation_worker_seconds=0.,scoring_wait_worker_seconds=0.))
    value=b.review(base,candidate)
    assert value['candidate']['steady_collections']==3 and len(value['candidate']['invocation_transitions'])==16
    assert value['matched_cycle_end_updates']==[1506,1508,1510]
    assert value['matched_full_cycle_speedup']==pytest.approx(1.5)
    assert value['candidate']['timing_windows'][-1]['cross_invocation']is True
