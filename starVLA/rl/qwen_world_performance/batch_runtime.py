"""Configurable fresh multi-scene windows using the proven actor/scorer/trainer.

Every leaf is one genuine scene. All leaves are collected before normalization;
one backward accumulation, DDP reduction, clip and Adam step per inner update.
"""
import copy
import time
import traceback
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.distributed as dist

from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_safety_exploration.group_runtime import GroupRuntime, collect_group
from starVLA.rl.qwen_world_safety_exploration.credit import GroupCredit, canonical_components
from starVLA.rl.qwen_world_research.runtime import ResearchRuntime
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from .contracts import validate_recipe, SCHEMA, ROOT
from .stage_scheduler import SceneExposureScheduler, stage_for_epoch


class WindowConditions:
    def __init__(self, base, capacity=32):
        self.base, self.capacity, self.cache = base, capacity, OrderedDict()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def materialize(self, descriptor, device):
        token = descriptor['token']
        if token in self.cache:
            self.base.token = token
            self.base.cached = checkpoint.device_tree(self.cache.pop(token), device)
        result = self.base.materialize(descriptor, device)
        self.cache[token] = checkpoint.cpu_tree(self.base.cached)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return result


class ComponentCapture:
    def record_policy_components(self, pg, anchor):
        self.pg_component, self.weighted_anchor_component = pg, anchor


class PerformanceLoss(ComponentCapture, ResearchLoss):
    pass


class HistoricalBudgetMixin:
    """One proactive historical scene/rank, within the same beta50 budget."""
    history_share=0.

    def _policy_loss(self,model,rollout):
        if rollout.get('performance_history_only'):
            from .persistent_history import history_loss
            value,metrics=history_loss(model,rollout['material'],self.auxiliary_state)
            if value is None:
                # A genuine current fresh-condition graph gives every rank one
                # final DDP forward/backward, even on empty or dominated banks.
                value,_=ResearchLoss._policy_loss(self,model,rollout['fallback'])
                value=value*0.
            return value,metrics
        total,metrics=super()._policy_loss(model,rollout)
        if self.history_share:
            total=total-self.history_share*self.weighted_anchor_component
        return total,metrics


class PerformanceRuntime(GroupRuntime):
    recipe_validator = staticmethod(validate_recipe)

    @staticmethod
    def add_arguments(parser):
        ResearchRuntime.add_arguments(parser)
        parser.add_argument('--qualification-force-auxiliary',action='store_true',
            help='QUALIFICATION ONLY: exercise rescue at IL0 and own-history at second collection; never a formal run')

    def load_recipe(self,args):
        if getattr(args,'qualification_force_auxiliary',False) and not args.qualification:
            raise ValueError('Artificial auxiliary gate is forbidden in formal training')
        return super().load_recipe(args)

    @staticmethod
    def source_files():
        files = GroupRuntime.source_files()
        for folder in ('starVLA/rl/qwen_world_performance', 'tools/qwen_world_performance'):
            for p in Path(folder).rglob('*.py'):
                files[str(p)] = file_sha(p)
        files['tools/qwen_world_safety_exploration/train.py'] = file_sha('tools/qwen_world_safety_exploration/train.py')
        return dict(sorted(files.items()))

    def make_credit(self, cfg):
        self.spec = cfg['research']['performance']
        credit = GroupCredit(self.spec['group_size'])
        credit.protocol.update(schema='world_performance_safetyep_v1',
            normalization='population std of all VALID fresh scene raw advantages, including zero groups; padding excluded',
            inner_updates=self.spec['inner_updates'])
        return credit

    def make_scene_stream(self, cfg, args, world):
        from .sampling_priority import PerformanceSceneStream
        self.stream = PerformanceSceneStream(args.manifest, world)
        self.stream.set_batch_size(self.spec['scenes_per_collection'])
        return self.stream

    def make_actor(self, cfg, args, device):
        actor, a, b = super().make_actor(cfg, args, device)
        self.conditions = WindowConditions(self.conditions)
        object.__setattr__(actor, 'conditions', self.conditions)
        alg = self.spec['algorithm']
        loss_type = PerformanceLoss
        if alg == 'awfm':
            from .awfm import AWFMLoss
            loss_type = type('CapturedAWFM', (ComponentCapture, AWFMLoss), {})
        elif alg == 'fpo':
            from .fpo_adapter import FPOAdapterLoss
            loss_type = type('CapturedFPO', (ComponentCapture, FPOAdapterLoss), {})
        elif alg in ('nominal_rescue','temporal_rescue','performance_max'):
            from .nominal_improvement import NominalLossMixin
            mixins=(HistoricalBudgetMixin,) if alg=='performance_max' else ()
            loss_type = type('CapturedNominal', mixins+(NominalLossMixin,ComponentCapture,ResearchLoss), {})
        actor._loss = loss_type(cfg)
        return actor, a, b

    def make_scheduler(self, optimizer, cfg):
        self.scheduler = SceneExposureScheduler(optimizer, self.spec['schedule'], self.spec['population'])
        first=self.spec['stages'][0]
        self.scheduler.set_stage(first['name'],[first['action_lr'],first['memory_lr']])
        return self.scheduler

    def bind_training_state(self, state, rollout, identity):
        super().bind_training_state(state, rollout, identity)
        if 'performance_rank' in state and state['performance_rank']!=dist.get_rank():
            raise ValueError('Native rank-private bank/RNG belongs to another rank')
        state['performance_rank']=dist.get_rank()
        for k, value in dict(valid_scene_exposures=0, padded_scene_exposures=0, collection_ledger=[],
                extra_B1_queries=0,adaptive_G_queries=0,history_B1_queries=0,seconds_auxiliary=0.,
                performance_auxiliary={}, evaluation_milestones=[], stage_transitions=[], current_stage=self.spec['stages'][0]['name']).items():
            if k not in state:
                if state['updates'] or state['collections']:
                    raise ValueError('Missing native performance state cannot be reconstructed silently')
                state[k] = copy.deepcopy(value)
        self.stream.run_identity=identity['sha256']
        self.stream.bind(state)
        if self.spec['algorithm'] in ('nominal_rescue','temporal_rescue','performance_max'):
            from .nominal_improvement import new_calibration_state
            if (state['updates'] or state['collections']) and 'nominal_improvement' not in state['performance_auxiliary']:
                raise ValueError('Nonfresh native run is missing its auxiliary calibration state')
            self.actor._loss.auxiliary_state=state['performance_auxiliary'].setdefault('nominal_improvement',new_calibration_state())
        if self.spec['algorithm']=='performance_max':
            from .persistent_history import new_history_state
            if (state['updates'] or state['collections']) and 'history' not in state['performance_auxiliary']:
                raise ValueError('Nonfresh native S5 cannot recreate a lost history bank')
            state['performance_auxiliary'].setdefault('history',new_history_state(identity['sha256'],dist.get_rank()))
        self.stage = next(copy.deepcopy(s) for s in self.spec['stages'] if s['name']==state['current_stage'])
        self.stream.set_batch_size(self.stage['B'])

    def _stage_boundary(self, actor, version):
        state = self.training_state
        target = stage_for_epoch(self.spec, state['valid_scene_exposures']/self.spec['population'])
        if target['name'] != self.stage['name']:
            old = self.stage
            self.stream.set_batch_size(target['B'])
            self.scheduler.set_stage(target['name'], [target['action_lr'], target['memory_lr']])
            receipt = dict(old_stage=old, new_stage=target, old_config_sha=fingerprint(old), new_config_sha=fingerprint(target),
                current_model_sha=version['adapter_sha256'],pinned_parent_sha=self.parent['model_sha256'],
                model_hash_definition='trainable LoRA+memory state SHA; complete model also requires pinned parent SHA',
                optimizer_restored=bool(self.args.resume),optimizer_state_preserved=True,
                optimizer_reset=False, scheduler_transition=self.scheduler.state_dict(),
                scene_exposures=state['valid_scene_exposures'], candidate_queries=state['new_candidates'])
            state['stage_transitions'].append(receipt)
            error=None
            if dist.get_rank() == 0:
                try:
                    from starVLA.rl.flow_grpo.transactions import atomic_json
                    destination=Path(self.args.output)/'stage_transitions'/target['name']/'stage_transition_receipt.json'
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    atomic_json(destination, receipt)
                except Exception:error=traceback.format_exc()
            checkpoint.check_errors(error)
            self.stage = target
            state['current_stage']=target['name']

    def before_collection(self,actor,state):
        target=stage_for_epoch(self.spec,state['valid_scene_exposures']/self.spec['population'])
        if target['name']!=self.stage['name']:
            from tools.qwen_world_safety_exploration.train import parameter_hash
            self._stage_boundary(actor,dict(adapter_sha256=parameter_hash(actor,trainable=True)))

    @torch.no_grad()
    def collect(self, actor, condition, token, log, service, generator, *, variant, version, progress=None):
        self._stage_boundary(actor, version)
        rank = dist.get_rank(); world = dist.get_world_size()
        start = self.training_state['scene_cursor']; b = self.stage['B']//world
        samples, positions, states = [], [], [generator.get_state().clone()]
        for micro in range(b):
            t, position = self.stream.at(start+world*micro, rank)
            cond = condition if micro == 0 and t == token else self.conditions.get(t, actor.device)
            leaf = collect_group(actor, cond, t, self.stream.logs[t], service, generator,
                group_size=self.group_size, variant=variant, version=version, progress=progress)
            leaf['stream_position'] = position
            leaf['valid_scene'] = position.get('valid', True)
            samples.append(leaf); positions.append(position); states.append(generator.get_state().clone())
            if leaf['valid_scene']:
                aux_started=time.monotonic()
                self._attach_auxiliary(actor, leaf, service, rank)
                leaf['auxiliary_seconds']=time.monotonic()-aux_started
        window = dict(schema=SCHEMA, samples=samples, rank=rank, group_size=self.group_size,
            B=self.stage['B'], M=self.spec['inner_updates'], window_start_cursor=start,
            window_positions=positions, generator_states=states,
            cond_bg=dict(schema='world_performance_window_conditions_v1', observations=[s['cond_bg'] for s in samples]),
            chain=torch.stack([s['chain'] for s in samples]), old_logp=torch.stack([s['old_logp'] for s in samples]),
            token=samples[0]['token'], log=samples[0]['log'], tokens=[s['token'] for s in samples],
            logs=[s['log'] for s in samples], behavior_version=copy.deepcopy(version),
            records=[r for s in samples for r in s['records']], rewards=torch.cat([s['rewards'] for s in samples]),
            protocol_sha256=service.protocol['sha256'],
            seconds={k:sum(s['seconds'][k] for s in samples) for k in samples[0]['seconds']})
        if self.spec['algorithm']=='performance_max':
            from .persistent_history import admit_nominal,attach_history_window
            bank=self.training_state['performance_auxiliary']['history']
            for sample in samples:
                nominal=sample.get('nominal_improvement',{}).get('nominal')
                if nominal is not None:admit_nominal(bank,nominal)
            force_history=getattr(self.args,'qualification_force_auxiliary',False) and self.training_state['collections']>=1
            if self.stage['history'] or force_history:
                history_started=time.monotonic()
                window['performance_history']=attach_history_window(bank,actor,self.conditions,service,version)
                window['history_seconds']=time.monotonic()-history_started
        local_queries=[sum(s.get('nominal_improvement',{}).get(k,0) for s in samples)
            for k in ('actual_B1_queries','actual_extra_teacher_queries')]
        local_queries.append(window.get('performance_history',{}).get('actual_B1_queries',0))
        window['auxiliary_queries_local']=local_queries
        window['auxiliary_seconds_local']=sum(s.get('auxiliary_seconds',0.) for s in samples)+window.get('history_seconds',0.)
        return window

    def _attach_auxiliary(self, actor, leaf, service, rank):
        alg = self.spec['algorithm']
        force=getattr(self.args,'qualification_force_auxiliary',False)
        if self.stage['rescue'] or force:
            from .nominal_improvement import attach_material
            attach_material(actor,leaf,self.conditions,service,
                exposure_epoch=(.5 if force else self.training_state['valid_scene_exposures']/self.spec['population']),rank=rank)
        # Nominal rescue and persistent history are attached below by their
        # validated adapters. Extra trajectories never change this fresh leaf.

    def validate_collection(self, window):
        if len(window['samples'])*8 != window['B']:
            raise ValueError('Wrong number of different scenes per fresh policy version')
        valid = [s['token'] for s in window['samples'] if s['valid_scene']]
        if len(set(valid)) != len(valid):
            raise ValueError('A collection must contain distinct valid scenes per rank')
        for leaf in window['samples']:
            if leaf['stoch_idx'].tolist() != [7,8,9] or leaf['chain'].shape != (self.group_size,11,8,3):
                raise ValueError('Source W3/candidate chain changed')
            if self.args.qualification and (leaf['qualification_sampling_check'] or {}).get('status')!='PASS':
                raise ValueError('Actual source sample and recomputed transition means/std must agree exactly')
            canonical_components(leaf['records'],self.group_size,device=leaf['rewards'].device)

    @torch.no_grad()
    def finalize_advantages(self, window):
        samples = window['samples']; g = self.group_size
        counts=torch.tensor(window['auxiliary_queries_local'],device=window['rewards'].device,dtype=torch.int64)
        dist.all_reduce(counts);window['auxiliary_queries_global']=counts.cpu().tolist()
        components = torch.cat([canonical_components(s['records'],g,device=s['rewards'].device)['components'] for s in samples])
        rewards = window['rewards']
        local_material = torch.cat((rewards.double().unsqueeze(-1),components),-1)
        gathered = [torch.empty_like(local_material) for _ in range(8)]
        dist.all_gather(gathered,local_material)
        valid = torch.tensor([s['valid_scene'] for s in samples],device=rewards.device,dtype=torch.bool)
        masks=[torch.empty_like(valid) for _ in range(8)];dist.all_gather(masks,valid)
        population=torch.cat(gathered)[torch.cat(masks)]
        stats=self.credit.advantages(rewards,components,all_material=population)
        from .metrics import window_counts
        window['quality_counts_global']=window_counts(window)
        if self.spec['algorithm']=='temporal_rescue':
            from .temporal_credit import temporal_credit_parts
            parts=temporal_credit_parts(components,[s['records'] for s in samples],base_parts=self.credit.parts(components))
            raw=parts['A_raw'];all_raw=[torch.empty_like(raw) for _ in range(8)];dist.all_gather(all_raw,raw)
            scale=torch.cat(all_raw)[torch.cat(masks)].std(unbiased=False)
            stats.update({k:parts[k] for k in ('feasible','cost','A_safe','A_EP','A_raw')})
            stats.update(normalization_std=scale,advantages=(raw/(scale+1e-8)).reshape(-1).float(),
                adv_mask=raw.ne(0).any(1,keepdim=True).expand_as(raw).reshape(-1).float())
            window['temporal_audit']=parts['temporal_audit']
            stats['statistics']['structured_zero_groups']=int((~raw.ne(0).any(1)).sum())
            stats['statistics']['effective_PG_candidate_fraction']=float(stats['adv_mask'].mean())
        window.update(valid_global_scenes=int(torch.cat(masks).sum()),
            advantages=stats['advantages'],adv_mask=stats['adv_mask'],global_std=stats['global_std'],
            normalization_std=stats['normalization_std'],advantage_statistics=stats['statistics'])
        for i,s in enumerate(samples):
            sl=slice(i*g,(i+1)*g)
            s.update(advantages=stats['advantages'][sl],adv_mask=stats['adv_mask'][sl]*valid[i],
                normalization_std=stats['normalization_std'],global_std=stats['global_std'],
                raw_group_std=stats['raw_group_std'][i:i+1],advantage_statistics=copy.deepcopy(stats['statistics']),
                structured_advantage={k:stats[k][i:i+1] for k in ('feasible','cost','A_safe','A_EP','A_raw')})
            window['adv_mask'][sl]*=valid[i]
            if self.spec['algorithm']=='awfm':
                from .awfm import attach_awfm_material
                attach_awfm_material(self.actor,s,rank=dist.get_rank())
            elif self.spec['algorithm']=='fpo':
                from .fpo_adapter import attach_fpo_material
                attach_fpo_material(self.actor,s,rank=dist.get_rank())
        return window

    def commit_collection_accounting(self, window, state, world):
        b,g,m=window['B'],window['group_size'],window['M'];valid=window['valid_global_scenes']
        state['collection_ledger'].append(dict(B=b,G=g,M=m,valid=valid,start_update=state['updates'],stage=self.stage['name']))
        state['collections']+=1;state['raw_scene_exposures']+=b;state['new_candidates']+=b*g;state['scene_cursor']+=b
        state['valid_scene_exposures']+=valid;state['padded_scene_exposures']+=b-valid
        for key,value in zip(('extra_B1_queries','adaptive_G_queries','history_B1_queries'),window['auxiliary_queries_global']):
            state[key]+=value
        state['seconds_auxiliary']+=window['auxiliary_seconds_local']
        self.scheduler.set_exposures(state['valid_scene_exposures'])

    def on_collection_committed(self, window, state):
        positions=checkpoint.gather(window['window_positions'])
        self.stream.commit_window([p for rank in positions for p in rank],state['scene_cursor'])
        from .feedback import accumulate_nominal_feedback,flush_feedback
        accumulate_nominal_feedback(state,window)
        flush_feedback(self,window,state)

    def boundary_phase(self, state, rollout):
        if state['updates']==self.cfg['max_steps']:return 'COMPLETE'
        if rollout is None:return 'READY'
        return 'INNER_COMMITTED' if state['next_inner'] else 'ROLLOUT_COMMITTED'

    def evaluation_due(self, state):
        epoch=state['valid_scene_exposures']/self.spec['population']
        if state['next_inner']:return False
        new=[e for e in self.spec['evaluation_epochs'] if e<=epoch and e not in state['evaluation_milestones']]
        state['evaluation_milestones'].extend(new)
        return bool(new)

    def backward_update(self, model, actor, window, state):
        losses, rows, errors = [], [], []
        weights=[]
        n=len(window['samples'])
        proactive='performance_history' in window
        hshare=window['performance_history']['hshare'] if proactive else 0.
        if hshare<0 or hshare+.05>1:raise ValueError('IL/rescue/history beta-budget shares must remain nonnegative')
        actor._loss.history_share=hshare
        for index,leaf in enumerate(window['samples']):
            with (model.no_sync() if (index<n-1 or proactive) and hasattr(model,'no_sync') else nullcontext()):
                failure=None
                try:
                    actor._performance_fresh_inner=state['next_inner']
                    total,metrics=model(leaf)
                    current=actor._probability_audit['current']
                    ratio=(current['source_clamped_logprob'].mean((-1,-2))-leaf['old_logp']).exp()
                    error=float((ratio-1).abs().max())
                    if not torch.isfinite(total) or not torch.isfinite(ratio).all():raise FloatingPointError('Nonfinite fresh window')
                    if state['next_inner']==0 and error>1e-4:raise ValueError('Collection contains a stale source ratio')
                except Exception:failure=traceback.format_exc()
                if failure and checkpoint.fatal_device_error(failure):raise RuntimeError(failure)
                checkpoint.check_errors(failure)
                weight=8/window['valid_global_scenes'] if leaf['valid_scene'] else 0.
                (total*weight).backward()
            losses.append(total.detach());rows.append({k:v.detach() for k,v in metrics.items()});errors.append(error)
            weights.append(weight)
        zero=total.detach().new_zeros(())
        metrics={k:torch.stack([r.get(k,zero) for r in rows]).mean() for k in set().union(*(r.keys() for r in rows))}
        active=[r for r in rows if float(r.get('nominal_aux_enabled',zero))>0]
        metrics['nominal_active_teacher_slots_local']=zero+len(active)
        for k in ('nominal_scale_cfm','nominal_scale_ode','nominal_aux_raw_cfm','nominal_aux_raw_ode',
                'nominal_aux_teacher_gain','nominal_normed_cfm','nominal_normed_ode'):
            if active:metrics[k+'_conditional_mean']=torch.stack([r[k] for r in active]).mean()
        calibration=state['performance_auxiliary'].get('nominal_improvement')
        if calibration:metrics['nominal_calibration_effective_updates']=zero+calibration['effective_updates']
        history_total=total.detach().new_zeros(())
        if proactive:
            payload=dict(performance_history_only=True,material=window['performance_history'],fallback=window['samples'][0])
            failure=None
            try:history_value,history_metrics=model(payload)
            except Exception:failure=traceback.format_exc()
            if failure and checkpoint.fatal_device_error(failure):raise RuntimeError(failure)
            checkpoint.check_errors(failure)
            local_valid=sum(int(s['valid_scene']) for s in window['samples'])
            history_total=50*hshare*history_value*(local_valid*8/window['valid_global_scenes'])
            history_total.backward()
            metrics.update({k:v.detach() for k,v in history_metrics.items()})
        metrics.update(valid_scene_exposures=total.new_tensor(state['valid_scene_exposures']),
            exposure_epoch=total.new_tensor(state['valid_scene_exposures']/self.spec['population']),
            global_distinct_scenes=total.new_tensor(window['B']), fresh_PDM_queries=total.new_tensor(window['B']*self.group_size))
        metrics.update({k:total.new_tensor(v) for k,v in window['quality_counts_global'].items()})
        return sum((v*w for v,w in zip(losses,weights)),zero)+history_total.detach(),metrics,dict(fresh_error=max(errors),source_clamp_hit_rate=float(current['source_clamp_hit_rate']))

    def exploration_summary(self, window):
        return dict(schema=SCHEMA,B=window['B'],G=window['group_size'],M=window['M'],
            tokens=window['tokens'],valid_scene_count=window['valid_global_scenes'],
            normalization_candidates=window['valid_global_scenes']*self.group_size,
            normalization_std=float(window['normalization_std']),
            auxiliary_queries_global=window['auxiliary_queries_global'],
            quality_counts_global=window['quality_counts_global'],
            raw_zero_groups=sum(not bool(s['adv_mask'].any()) for s in window['samples']))

    def recipe_identity(self,cfg,args,condition_sha):
        value=ResearchRuntime.recipe_identity(self,cfg,args,condition_sha)
        from .nominal_improvement import CONFIG as NOMINAL_CONFIG,CALIBRATION_PROTOCOL
        from .persistent_history import PROTOCOL as HISTORY_PROTOCOL
        from .awfm import PROTOCOL as AWFM_PROTOCOL
        from .fpo_adapter import PROTOCOL as FPO_PROTOCOL
        from .temporal_credit import PROTOCOL as TEMPORAL_PROTOCOL
        value.update(fresh_IL70=True, NAVTEST_SELECTED=True, performance_protocol=copy.deepcopy(self.spec),
            qualification_auxiliary_gate_override=bool(getattr(args,'qualification_force_auxiliary',False)),
            performance_implementation_protocols=dict(nominal=NOMINAL_CONFIG,calibration=CALIBRATION_PROTOCOL,
                history=HISTORY_PROTOCOL,awfm=AWFM_PROTOCOL,fpo=FPO_PROTOCOL,temporal=TEMPORAL_PROTOCOL,
                stage_learning_rate='stage per-group peaks multiplied by the same global exposure schedule; LR floor is not restarted at stage transitions'),
            exploration_protocol=dict(B=self.spec['scenes_per_collection'],G=self.group_size,M=self.spec['inner_updates'],
                stochastic_steps=[7,8,9], ratio='source transition proxy; not endpoint density',
                scene_loss='equal weight per valid scene; padding has zero PG and anchor',
                normalization='all fresh valid B*G candidates in one behavior version'),
            initialization='complete identical IL70; all trainables fresh LoRA; Adam0; no cross-algorithm state')
        return value

    def finalize_training_identity(self,identity):
        identity['scene_reduction']='equal weight per valid fresh scene over all ranks; padding zero; global valid-count denominator'
        identity['actual_optimizer'].update(warmup_updates=None,
            warmup_valid_scene_epochs=self.spec['schedule']['warmup_epochs'],
            schedule_clock='valid_scene_exposures; repeated Adam inners do not advance the scene clock',
            registered_schedule=copy.deepcopy(self.spec['schedule']),
            initial_actual_group_lrs=list(self.scheduler._last_lr),
            lr_field_meaning='registered peak; actual LR is stage peaks times the global exposure multiplier',
            inner_updates_per_fresh_collection=self.spec['inner_updates'])

    def validate_boundary(self,*args,**kwargs):
        from .recovery import validate_boundary
        return validate_boundary(*args,**kwargs)

    def validate_ranks(self,summaries):
        from .recovery import validate_ranks
        return validate_ranks(summaries)

    def public_training_state(self,state):
        value=dict(state)
        value['exposure_epoch']=state['valid_scene_exposures']/self.spec['population']
        for k in ('collection_ledger','scene_stream','performance_auxiliary'):
            if k in value:value[k]=dict(material_stored_in_native_checkpoint=True)
        return value
