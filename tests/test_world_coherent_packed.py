"""Coherent source chains, per-scene credit, pending replay and cache identity."""
import copy,io
from types import SimpleNamespace
import pytest,torch,yaml
from test_world_multiscene_prototype import make
from test_world_performance_nominal import Service
from starVLA.rl.qwen_world_multiscene.collection import collect_packed
from starVLA.rl.qwen_world_multiscene.conditions import PackedConditions
from starVLA.rl.qwen_world_multiscene.objective import source_probabilities,source_loss
from starVLA.rl.qwen_world_multiscene.cache import ImmutableFeatureCache
from starVLA.rl.qwen_world_multiscene.runtime import FastPackedRuntime,PackedSourceLoss
from starVLA.rl.qwen_world_multiscene.recovery import validate_packs
from starVLA.rl.qwen_world_multiscene.recipe import build_recipe,validate_recipe
from starVLA.rl.qwen_world_performance.batch_runtime import WindowConditions
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit,canonical_components
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest


def setup(count):
    a,conditions,old,_=make(count);conditions=ImmutableFeatureCache(conditions);object.__setattr__(a,'conditions',conditions)
    tokens=conditions.data.tokens;service=Service();generator=torch.Generator().manual_seed(42)
    valid=[True]*count;valid[-1]=False
    leaves,pack=collect_packed(a,conditions,[conditions.get(t,a.device)for t in tokens],tokens,['TRAIN_log']*count,
        service,generator,version=dict(run='coherent_CPU',update=0,adapter_sha256='same'),pack_index=0,valid_masks=valid)
    assert service.counts==[16]*count
    material=torch.cat([torch.cat((s['rewards'].double().unsqueeze(-1),canonical_components(s['records'],16)['components']),-1)for s in leaves])
    population=material[torch.tensor(valid)].repeat(8,1,1)
    for i,s in enumerate(leaves):
        stats=GroupCredit(16).advantages(s['rewards'],canonical_components(s['records'],16)['components'],all_material=population)
        s.update(advantages=stats['advantages'],adv_mask=stats['adv_mask']*valid[i],valid_scene=valid[i])
    window=dict(samples=leaves,packed_groups=[pack],packed_execution=dict(micro_scenes=count,candidate_group=16),
        valid_global_scenes=sum(valid)*8,B=count*8,quality_counts_global={})
    runtime=FastPackedRuntime();runtime.conditions=conditions;runtime.group_size=16;runtime.spec=dict(microbatch_scenes=count,population=103288)
    a._loss=PackedSourceLoss(a._loss.cfg)if hasattr(a._loss,'cfg')else a._loss
    #ResearchLoss does not retain cfg; same original Scope recipe for this tiny actor.
    from test_world_safety_exploration_groups import cfg
    a._loss=PackedSourceLoss(cfg(16))
    return a,conditions,runtime,window,generator


@pytest.mark.parametrize('count',[2,4])
def test_coherent_pack_fresh_fourinner_exact_pending_resume(count,monkeypatch):
    torch.set_num_threads(1);a,cond,rt,window,generator=setup(count);state=dict(next_inner=0,valid_scene_exposures=8*(count-1))
    validate_packs(window,window['packed_execution'])
    initial={k:v.clone()for k,v in a.state_dict().items()};opt=torch.optim.AdamW([p for p in a.parameters()if p.requires_grad],lr=1e-5)
    for inner in range(2):
        state['next_inner']=inner;opt.zero_grad(set_to_none=True);_,_,audit=rt.backward_update(a,a,window,state)
        if inner==0:assert audit['fresh_error']==0
        opt.step()
    data=io.BytesIO();torch.save(dict(actor=a.state_dict(),optimizer=opt.state_dict(),window=window,generator=generator.get_state()),data)
    for inner in range(2,4):
        state['next_inner']=inner;opt.zero_grad(set_to_none=True);rt.backward_update(a,a,window,state);opt.step()
    b,bc,br,bw,bg=setup(count);data.seek(0);saved=torch.load(data,weights_only=False)
    b.load_state_dict(saved['actor']);bo=torch.optim.AdamW([p for p in b.parameters()if p.requires_grad],lr=1e-5);bo.load_state_dict(saved['optimizer']);bg.set_state(saved['generator']);bw=saved['window']
    validate_packs(bw,bw['packed_execution'])
    for inner in range(2,4):
        state['next_inner']=inner;bo.zero_grad(set_to_none=True);br.backward_update(b,b,bw,state);bo.step()
    assert all(torch.equal(v,b.state_dict()[k])for k,v in a.state_dict().items())
    assert digest(opt.state_dict())==digest(bo.state_dict())and torch.equal(generator.get_state(),bg.get_state())
    bad=copy.deepcopy(bw);bad['samples'][0]['packed_position']['scene_ordinal']=1
    with pytest.raises(ValueError,match='reordered'):validate_packs(bad,bw['packed_execution'])
    bad=copy.deepcopy(bw);bad['packed_groups'][0]['sampled_mean'][0,0,0,0]+=.1
    with pytest.raises(ValueError,match='means'):validate_packs(bad,bw['packed_execution'])


def test_immutable_cache_hits_copy_zero_and_preserve_values_gradients(monkeypatch):
    torch.set_num_threads(1);a,base,samples,_=make();old=WindowConditions(base);new=ImmutableFeatureCache(base)
    desc=[s['cond_bg']for s in samples]
    x=PackedConditions(old,selected_layers=a.scene_provider.selected_layers).pack(desc,a.device)
    y=PackedConditions(new,selected_layers=a.scene_provider.selected_layers).pack(desc,a.device)
    assert digest(x)==digest(y);before=new.statistics();storage=[new.device_cache[t]['base_memory'].data_ptr()for t in base.data.tokens]
    for _ in range(3):
        z=PackedConditions(new,selected_layers=a.scene_provider.selected_layers).pack(desc,a.device);assert digest(z)==digest(y)
    after=new.statistics();assert before['host_backups']==after['host_backups']==2 and after['hits']-before['hits']==6
    assert storage==[new.device_cache[t]['base_memory'].data_ptr()for t in base.data.tokens]
    from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
    from test_world_safety_exploration_groups import cfg
    grads=[]
    for condition in (x,y):
        a.zero_grad(set_to_none=True);loss,_,_=source_loss(a,samples,condition,loss=ResearchLoss(cfg(16)));loss.backward()
        grads.append({n:p.grad.clone()for n,p in a.named_parameters()if p.requires_grad})
    assert digest(grads[0])==digest(grads[1])
    c=yaml.safe_load(open('/mnt/project/qwen-world-performance-rl-20261103/configs/S0_LARGE_SAFETYEP.yaml'))
    for count in (2,4):assert validate_recipe(build_recipe(c,count))['research']['performance']['microbatch_scenes']==count
    c=build_recipe(c,2);c['research']['performance']['algorithm']='nominal_rescue'
    with pytest.raises(ValueError,match='without auxiliary'):validate_recipe(c)
    #Actual complete READY validator/stream/scheduler path (no synthetic Adam
    #moments or migrated source identity); packed protocol is mandatory.
    from starVLA.rl.qwen_world_multiscene.recovery import validate_boundary
    c=yaml.safe_load(open('/mnt/project/qwen-world-performance-rl-20261103/configs/S0_LARGE_SAFETYEP.yaml'));c=build_recipe(c,2)
    rt=FastPackedRuntime();rt.cfg=c;rt.spec=c['research']['performance'];rt.actor=a;rt.group_size=16
    args=SimpleNamespace(manifest='/mnt/project/qwen-world-performance-rl-20261103/data/NAVTRAIN_SCENES.json')
    stream=rt.make_scene_stream(c,args,8);optimizer=rt.make_optimizer(a,c);scheduler=rt.make_scheduler(optimizer,c)
    state=dict(updates=0,collections=0,raw_scene_exposures=0,new_candidates=0,next_inner=0,scene_cursor=0,phase='READY')
    ident=dict(recipe=c,sha256='CPU_coherent_ready',world=8,data_stream_identity=stream.identity,packed_execution=c['research']['packed_execution'])
    monkeypatch.setattr('starVLA.rl.qwen_world_multiscene.runtime.dist.get_rank',lambda:0)
    rt.bind_training_state(state,None,ident)
    assert validate_boundary(state,None,ident,optimizer,scheduler)['packed_execution']==c['research']['packed_execution']
    bad=copy.deepcopy(ident);bad.pop('packed_execution')
    with pytest.raises(ValueError,match='identity'):validate_boundary(state,None,bad,optimizer,scheduler)
