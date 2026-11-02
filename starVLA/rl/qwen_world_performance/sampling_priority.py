"""Full-pass first, TRAIN-only priority thereafter; immutable collection plans.

Call bind(training_state) after restore, then at(scene_cursor+micro*world,rank).
Commit once per freshly collected globalB window, before saving pending inners:
commit_window(positions_or_None,next_cursor). All four inners reuse that rollout.
Padding is scored only for shape/physical-query accounting and has valid=False.
"""
import copy,math,random
from pathlib import Path
from .data_protocol import load_manifest,load_feedback,read_sealed,file_sha,fingerprint,physical,DEFAULT_POLICY

STATE_SCHEMA='world_performance_scene_clock_state_v1'


class PerformanceSceneStream:
    def __init__(self,manifest,world,*,batch_size=None,run_identity=None,priority=True,expected_count=103288):
        self.path=Path(manifest);self.manifest,self.logs=load_manifest(manifest,expected_count)
        self.tokens=list(self.manifest['tokens']);self.world=int(world);self.n=len(self.tokens)
        self.initial_batch=int(batch_size or world);self._check_batch(self.initial_batch)
        self.batch_size=self.initial_batch;self.run_identity=run_identity;self.priority=bool(priority)
        self.sha=file_sha(manifest);self._state=None;self._orders={};self._feedback_cache={}
        self.mode='performance_priority'if priority else'uniform'
        self.identity=dict(schema=STATE_SCHEMA,manifest_sha256=self.sha,world=self.world,
            count=self.n,initial_global_batch=self.initial_batch,seed=42,priority=self.priority,
            policy=copy.deepcopy(DEFAULT_POLICY),clock='global scene slots; valid exposures separately counted',
            uniform_cursor='independent source randperm cursor, never incremented by priority or padding')
        self.identity['sha256']=fingerprint(self.identity)

    def _check_batch(self,b):
        if self.world<=0 or b<=0 or b%self.world:raise ValueError('globalB must be a positive multiple of world')

    def bind(self,state):
        if self.run_identity is None:
            self.run_identity=state.get('run_identity')
        if not self.run_identity:raise ValueError('Own run identity must be explicitly bound')
        if 'scene_stream'not in state:
            if state.get('scene_cursor',0)!=0:raise ValueError('Cannot initialize a new stream at a nonzero native cursor')
            state['scene_stream']=dict(schema=STATE_SCHEMA,identity_sha256=self.identity['sha256'],run_identity=self.run_identity,
                cursor=0,epoch=0,epoch_start_cursor=0,batch_size=self.initial_batch,uniform_cursor=0,
                valid_scene_exposures=0,padding_slots=0,priority_counts={},priority_unique_tokens=[],
                counters=dict(uniform=0,nominal_hard=0,regression=0,priority_fallback=0,priority_repeats=0,uniform_duplicates_skipped=0),
                epoch_history=[],
                feedback=None,pending_feedback=None,last_feedback_adoption_clock=None,selection_plan=None,last_commit=None)
        s=state['scene_stream']
        if(s['schema']!=STATE_SCHEMA or s['identity_sha256']!=self.identity['sha256']
           or s['run_identity']!=self.run_identity):raise ValueError('Native stream/run identity changed')
        self._state=s;self.batch_size=s['batch_size'];self._check_batch(self.batch_size)
        if s['cursor']%self.batch_size or s['cursor']<s['epoch_start_cursor']:
            raise ValueError('Invalid native scene cursor')
        if 'scene_cursor'in state and state['scene_cursor']!=s['cursor']:raise ValueError('Trainer/stream native cursor mismatch')
        for key in ('feedback','pending_feedback'):
            if s[key]:self._bank(s[key])
        if s['selection_plan']:
            self._verify_plan(s['selection_plan'])
        stream_state_validation(state,run_identity=self.run_identity,stream_identity=self.identity,tokens=self.tokens)
        return self

    def set_batch_size(self,b):
        b=int(b);self._check_batch(b)
        if self._state is None:self.batch_size=b;self.initial_batch=b;self.identity['initial_global_batch']=b;self.identity.pop('sha256');self.identity['sha256']=fingerprint(self.identity);return
        s=self._state
        if b==self.batch_size:return
        if s['selection_plan']or s['cursor']!=s['epoch_start_cursor']:
            raise ValueError('globalB may change only at an epoch boundary with no pending selection')
        self.batch_size=b;s['batch_size']=b

    @property
    def epoch_size(self):return math.ceil(self.n/self.batch_size)*self.batch_size

    def _bank(self,ref):
        p=ref['path'];actual=file_sha(p)
        if actual!=ref['file_sha256']:raise ValueError('Completed TRAIN feedback changed after admission')
        if actual not in self._feedback_cache:
            self._feedback_cache[actual]=load_feedback(p,logs=self.logs,run_identity=self.run_identity)
        return self._feedback_cache[actual]

    def offer_feedback(self,path):
        if self._state is None:raise ValueError('Bind stream before offering feedback')
        b=load_feedback(path,logs=self.logs,run_identity=self.run_identity)
        if b['observed_valid_scene_clock']>self._state['valid_scene_exposures']:
            raise ValueError('Feedback claims a future native scene clock')
        ref=dict(path=str(Path(path).resolve()),file_sha256=file_sha(path),sha256=b['sha256'],
                 observed_valid_scene_clock=b['observed_valid_scene_clock'])
        old=self._state['pending_feedback']or self._state['feedback']
        if old and ref['observed_valid_scene_clock']<old['observed_valid_scene_clock']:
            raise ValueError('TRAIN feedback clock cannot go backwards')
        self._state['pending_feedback']=ref
        return ref

    def _uniform_at(self,cursor):
        epoch,index=divmod(cursor,self.n)
        if epoch not in self._orders:
            import torch
            g=torch.Generator(device='cpu').manual_seed(42+epoch)
            self._orders[epoch]=torch.randperm(self.n,generator=g).tolist()
            if len(self._orders)>3:del self._orders[min(self._orders)]
        return self.tokens[self._orders[epoch][index]]

    def _verify_plan(self,p):
        b=dict(p);saved=b.pop('sha256')
        if fingerprint(b)!=saved or p['start_cursor']!=self._state['cursor']or len(p['items'])!=self.batch_size:
            raise ValueError('Changed/incomplete native selection plan')
        if any(x['token']not in self.logs for x in p['items']):raise ValueError('Forbidden planned token')

    def _plan(self):
        s=self._state
        if s is None:raise ValueError('Bind native state before stream access')
        if s['selection_plan']:self._verify_plan(s['selection_plan']);return s['selection_plan']
        offset=s['cursor']-s['epoch_start_cursor'];valid_clock=s['valid_scene_exposures'];feedback=s['feedback'];pending=s['pending_feedback']
        last=s['last_feedback_adoption_clock'];adopted=False
        if pending and s['epoch']>=1 and(last is None or valid_clock-last>=math.ceil(self.n*.5)):
            feedback=pending;adopted=True
        bank=self._bank(feedback)if feedback else None
        seed=int(fingerprint(dict(seed=42,manifest=self.sha,cursor=s['cursor'],feedback=feedback,uniform_cursor=s['uniform_cursor']))[:16],16)
        rng=random.Random(seed);uniform_cursor=s['uniform_cursor'];counts=dict(s['priority_counts']);unique=set(s['priority_unique_tokens']);counters=dict(s['counters']);items=[];used=set()
        # Protect all potential uniform/fallback positions in this collection.
        reserved={self._uniform_at(uniform_cursor+i)for i in range(self.batch_size*2)}
        def uniform():
            nonlocal uniform_cursor
            for _ in range(self.n+self.batch_size):
                t=self._uniform_at(uniform_cursor);uniform_cursor+=1
                if t not in used:return t
                counters['uniform_duplicates_skipped']+=1
            raise ValueError('Population too small for distinct valid globalB')
        pools={'nominal_hard':{},'regression':{}};observation_clocks={}
        if bank:
            for token,r in bank['records'].items():
                score=r['score'];drop=r.get('best_own_score',score)-score;metrics=r.get('metrics',{})
                observation_clocks[token]=r.get('observed_valid_scene_clock',bank['observed_valid_scene_clock'])
                log=physical(self.logs[token])
                if score<.9:pools['nominal_hard'].setdefault(log,[]).append((token,max(.05,1-score)))
                unsafe_now=any(metrics.get(k)!=1 for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))
                safety_regression=r.get('ever_strict_feasible',False)and unsafe_now
                high_regression=r.get('best_own_score',score)>=.9 and score<.9
                if drop>=.05 or safety_regression or high_regression:
                    pools['regression'].setdefault(log,[]).append((token,max(.05,drop+float(safety_regression))))
        def choose(kind,clock):
            groups={log:[(t,w)for t,w in rows if counts.get(t,0)<4 and t not in used and t not in reserved and 0<=clock-observation_clocks[t]<=self.n]
                    for log,rows in pools[kind].items()}
            groups={log:rows for log,rows in groups.items()if rows}
            if not groups:return None
            rows=groups[rng.choice(sorted(groups))];weights=[w for t,w in rows];r=rng.random()*sum(weights)
            for token,weight in rows:
                r-=weight
                if r<=0:return token
            return rows[-1][0]
        for i in range(self.batch_size):
            index=offset+i;valid=index<self.n;kind='uniform';requested='uniform'
            clock=valid_clock+min(i,max(0,self.n-offset))
            stale=bank is None or clock-bank['observed_valid_scene_clock']>self.n
            if valid and self.priority and s['epoch']>=1:
                k=index%20;requested='uniform'if k<12 else'nominal_hard'if k<17 else'regression'
            if not valid:token=self.tokens[index%self.n];kind='padding'
            elif requested!='uniform'and not stale:
                token=choose(requested,clock)
                if token is None:token=uniform();kind='uniform';counters['priority_fallback']+=1
                else:
                    kind=requested;counters['priority_repeats']+=int(token in unique);unique.add(token);counts[token]=counts.get(token,0)+1
            else:
                token=uniform()
                if requested!='uniform':counters['priority_fallback']+=1
            if valid:used.add(token);counters[kind]+=1
            items.append(dict(token=token,epoch=s['epoch'],index=index,global_position=s['cursor']+i,
                valid=valid,valid_mask=1 if valid else 0,padding=not valid,selection_kind=kind,
                requested_kind=requested,feedback_stale=stale,feedback_sha256=feedback['sha256']if feedback else None))
            items[-1]['feedback_observed_valid_scene_clock']=observation_clocks[token]if kind in('nominal_hard','regression')else None
        body=dict(start_cursor=s['cursor'],batch_size=self.batch_size,epoch=s['epoch'],items=items,
            next_uniform_cursor=uniform_cursor,next_priority_counts=counts,next_priority_unique_tokens=sorted(unique),
            next_counters=counters,feedback=feedback,adopt_pending_feedback=adopted,private_seed=seed,
            private_rng_state=rng.getstate())
        s['selection_plan']=dict(body,sha256=fingerprint(body));return s['selection_plan']

    def at(self,cursor,rank):
        if cursor<0 or cursor%self.world or not 0<=rank<self.world:raise ValueError('Invalid global micro-scene cursor/rank')
        p=self._plan();i=cursor+rank-p['start_cursor']
        if not 0<=i<self.batch_size:raise ValueError('Commit the previous full globalB window before accessing a new one')
        item=copy.deepcopy(p['items'][i]);token=item.pop('token');item['selection_plan_sha256']=p['sha256']
        return token,item

    def commit_window(self,positions,next_cursor):
        """positions may be None or the fullB at() metadata; returns next cursor.

        Commit only once after all fresh candidates/A have been finalized. Native
        state scene_cursor must advance to this returned value at the same point.
        """
        s=self._state;p=s.get('selection_plan')if s else None
        if p is None:raise ValueError('No planned window to commit; duplicate commit rejected')
        self._verify_plan(p)
        if next_cursor!=p['start_cursor']+self.batch_size:raise ValueError('Partial/skipped globalB commit')
        if positions is not None:
            if len(positions)!=self.batch_size:raise ValueError('Commit needs every rank/micro position')
            bypos={m['global_position']:m for m in positions}
            if len(bypos)!=self.batch_size:raise ValueError('Duplicate positions at commit')
            for x in p['items']:
                m=bypos.get(x['global_position'],{})
                if m.get('selection_plan_sha256')!=p['sha256']or any(m.get(k)!=x[k]for k in ('valid','padding','epoch','index','selection_kind')):
                    raise ValueError('Saved rollout/selection plan mismatch')
        nvalid=sum(x['valid']for x in p['items']);s['cursor']=next_cursor;s['valid_scene_exposures']+=nvalid;s['padding_slots']+=self.batch_size-nvalid
        for k in ('uniform_cursor','priority_counts','priority_unique_tokens','counters'):s[k]=p['next_'+k]
        s['feedback']=p['feedback']
        if p['adopt_pending_feedback']:
            if s['pending_feedback']==p['feedback']:s['pending_feedback']=None
            s['last_feedback_adoption_clock']=s['valid_scene_exposures']-nvalid
        s['last_commit']=dict(start_cursor=p['start_cursor'],next_cursor=next_cursor,plan_sha256=p['sha256'],valid=nvalid,padding=self.batch_size-nvalid,epoch=s['epoch'])
        s['selection_plan']=None
        if next_cursor-s['epoch_start_cursor']==self.epoch_size:
            s['epoch_history'].append(dict(epoch=s['epoch'],global_batch=self.batch_size,slots=self.epoch_size,valid=self.n))
            s['epoch']+=1;s['epoch_start_cursor']=next_cursor;s['priority_counts']={}
        return next_cursor

    def summary(self):
        s=self._state
        return dict(cursor=s['cursor'],epoch=s['epoch'],batch_size=self.batch_size,
            epoch_size=self.epoch_size,valid_scene_exposures=s['valid_scene_exposures'],padding_slots=s['padding_slots'],
            uniform_cursor=s['uniform_cursor'],priority_unique=len(s['priority_unique_tokens']),counters=s['counters'],
            pending_plan_sha256=s['selection_plan']['sha256']if s['selection_plan']else None)


def stream_state_validation(training_state,*,run_identity,stream_identity,tokens=None):
    """Pure native validator: no files, model, sampler RNG, or NAVTEST IDs read.

    Root boundary/rank validators should merge the returned digest into their
    summary. All ranks must agree; offer/broadcast a new TRAIN bank synchronously.
    `tokens` is the already validated legal NAVTRAIN stream membership, optional.
    """
    s=training_state['scene_stream'];n=stream_identity['count'];world=stream_identity['world'];b=s['batch_size']
    def require(ok,msg):
        if not ok:raise ValueError('Native scene stream: '+msg)
    require(s['schema']==STATE_SCHEMA and s['identity_sha256']==stream_identity['sha256']and s['run_identity']==run_identity,'identity changed')
    require(isinstance(b,int)and b>0 and b%world==0,'invalid globalB')
    require(s['epoch']>=0 and len(s['epoch_history'])==s['epoch'],'epoch history changed')
    for i,h in enumerate(s['epoch_history']):
        require(h['epoch']==i and h['global_batch']>0 and h['global_batch']%world==0 and h['slots']==math.ceil(n/h['global_batch'])*h['global_batch']and h['valid']==n,'epoch padding/history changed')
    require(sum(h['slots']for h in s['epoch_history'])==s['epoch_start_cursor'],'epoch start clock changed')
    offset=s['cursor']-s['epoch_start_cursor'];epoch_size=math.ceil(n/b)*b
    require(0<=offset<epoch_size and offset%b==0,'cursor not at a committed collection boundary')
    valid=s['epoch']*n+min(offset,n)
    require(s['valid_scene_exposures']==valid and s['padding_slots']==s['cursor']-valid,'valid/padding exposures changed')
    if 'scene_cursor'in training_state:require(training_state['scene_cursor']==s['cursor'],'trainer cursor differs')
    c=s['counters'];kinds=('uniform','nominal_hard','regression')
    require(all(isinstance(v,int)and v>=0 for v in c.values()),'negative/noninteger exposure counters')
    require(sum(c[k]for k in kinds)==valid,'exposure/category totals changed')
    require(s['uniform_cursor']==c['uniform']+c['uniform_duplicates_skipped'],'uniform cursor consumed by priority or padding')
    priority=c['nominal_hard']+c['regression'];unique=s['priority_unique_tokens']
    require(len(unique)==len(set(unique))and priority==len(unique)+c['priority_repeats'],'priority unique/repeat totals changed')
    require(all(isinstance(v,int)and 0<v<=4 for v in s['priority_counts'].values()),'per-token epoch priority cap violated')
    if s['epoch']==0:require(priority==0 and not s['priority_counts'],'first pass must remain uniform')
    legal=set(tokens)if tokens is not None else None
    if legal is not None:require(set(unique)<=legal and set(s['priority_counts'])<=legal,'forbidden priority token')
    for key in ('feedback','pending_feedback'):
        r=s[key]
        if r:
            require(r['observed_valid_scene_clock']<=valid and len(r['file_sha256'])==64 and len(r['sha256'])==64,'feedback clock/seal invalid')
    last=s['last_commit']
    if s['cursor']==0:require(last is None,'zero cursor has a commit')
    else:
        require(last is not None and last['next_cursor']==s['cursor']and last['next_cursor']>last['start_cursor'],'last commit identity changed')
        require(last['valid']+last['padding']==last['next_cursor']-last['start_cursor'],'last commit exposure mismatch')
        require(last['epoch']in(s['epoch'],s['epoch']-1)and len(last['plan_sha256'])==64,'last commit epoch/seal invalid')
    p=s['selection_plan']
    if p is not None:
        body=dict(p);saved=body.pop('sha256');require(fingerprint(body)==saved,'pending plan digest changed')
        require(p['start_cursor']==s['cursor']and p['epoch']==s['epoch']and p['batch_size']==b and len(p['items'])==b,'pending plan cursor/layout changed')
        expected_c=dict(c);expected_counts=dict(s['priority_counts']);expected_unique=set(unique);seen=set();uniform_count=0
        for i,x in enumerate(p['items']):
            isvalid=offset+i<n
            require(x['global_position']==s['cursor']+i and x['epoch']==s['epoch']and x['index']==offset+i,'planned scene position changed')
            require(x['valid']is isvalid and x['padding']is(not isvalid)and x['valid_mask']==int(isvalid),'padding contributes to valid exposure')
            if legal is not None:require(x['token']in legal,'forbidden planned token')
            if not isvalid:require(x['selection_kind']=='padding','invalid slot has training category');continue
            require(x['token']not in seen,'duplicate valid scene within globalB');seen.add(x['token'])
            kind=x['selection_kind'];require(kind in kinds,'unknown selection category');expected_c[kind]+=1
            if kind=='uniform':uniform_count+=1
            else:
                require(s['epoch']>=1 and p['feedback']is not None,'priority without own TRAIN feedback after full first pass')
                observed=x['feedback_observed_valid_scene_clock']
                require(isinstance(observed,int) and observed<=p['feedback']['observed_valid_scene_clock'],'invalid actual record observation clock')
                age=valid+i-observed;require(0<=age<=n,'stale/future priority feedback')
                expected_counts[x['token']]=expected_counts.get(x['token'],0)+1
                require(expected_counts[x['token']]<=4,'pending plan exceeds repetition cap')
                expected_c['priority_repeats']+=int(x['token']in expected_unique);expected_unique.add(x['token'])
        require(p['next_priority_counts']==expected_counts and set(p['next_priority_unique_tokens'])==expected_unique,'pending priority state changed')
        for k in kinds+('priority_repeats',):require(p['next_counters'][k]==expected_c[k],'pending exposure counters changed')
        require(p['next_uniform_cursor']>=s['uniform_cursor']+uniform_count,'pending uniform cursor skipped valid positions')
        require(p['next_uniform_cursor']==p['next_counters']['uniform']+p['next_counters']['uniform_duplicates_skipped'],'pending priority consumed uniform cursor')
    return dict(schema=STATE_SCHEMA,scene_cursor=s['cursor'],valid_scene_exposures=valid,
        padding_slots=s['padding_slots'],epoch=s['epoch'],global_batch=b,
        uniform_cursor=s['uniform_cursor'],stream_state_sha256=fingerprint(s),
        pending_selection_plan_sha256=p['sha256']if p else None)
