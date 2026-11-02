"""Read-only logical budgets, invocation costs, and completed B1 evaluations."""
import argparse
from pathlib import Path
import json
import time
from starVLA.rl.flow_grpo.transactions import atomic_json


def summarize(root):
    root=Path(root);runs={}
    for path in sorted((root/'runs').glob('*/identity.json')):
        run=path.parent;identity=json.loads(path.read_text());latest=json.loads((run/'latest.json').read_text())if(run/'latest.json').exists()else None
        state=None
        if latest:
            import torch
            own=torch.load(Path(latest['checkpoint'])/'rank0.pt',map_location='cpu',weights_only=False);state=own['state']
            if 'd6_history'in state:
                state=dict(state);history=state.pop('d6_history')
                state['d6_history_summary']={k:history[k]for k in ('schema','run','commits','digest')}
                state['d6_history_summary'].update(scene_count=len(history['rows']),teacher_count=len(history['teachers']))
        invocations=[json.loads(p.read_text())for p in (run/'invocations').glob('*/rank*.json')]
        committed=[json.loads(s)for s in(run/'metrics_rank0.jsonl').read_text().splitlines()]if(run/'metrics_rank0.jsonl').exists()else[]
        recent=committed[-20:];window=None
        if len(recent)>=3:
            dt=recent[-1]['at']-recent[0]['at'];du=recent[-1]['updates']-recent[0]['updates']
            window=dict(optimizer_updates_per_second=du/dt if dt>0 else None,from_update=recent[0]['updates'],to_update=recent[-1]['updates'])
        runs[run.name]=dict(identity=identity['sha256'],recipe=identity['recipe'],latest=latest,last_saved_state=state,
            last_journal_update=committed[-1]['updates']if committed else 0,recent_complete_window=window,
            invocation_count=len({x['id']for x in invocations}),physical_candidates_all_rank_attempts=sum(x.get('physical_candidates',0)for x in invocations),
            measured_finished_shared_worker_hours=sum(x.get('worker_wall_seconds',0)for x in invocations)/3600,
            checkpoint_save_worker_seconds=sum(x.get('seconds_save',0)for x in invocations),
            cost_limitations='Worker wall is shared occupancy time, not exclusive GPU compute. RUNNING cost is incomplete; logical counters come from latest legal checkpoint.')
    evaluations=[]
    for path in (root/'evaluations').glob('**/SUMMARY.json'):
        value=json.loads(path.read_text());evaluations.append(dict(path=str(path),summary=value))
    required={'D0','D1','D2','D3_FPO','D3_AWM','D4','D5','D6'}
    finished={r['recipe']['arm']for r in runs.values()if r['last_saved_state']and r['last_saved_state']['valid_scene_exposures']>=r['recipe']['target_epochs']*103288}
    return dict(schema='d44_campaign_status_v1',at=time.time(),runs=runs,evaluations=evaluations,
        parent_audit=str(root/'parent_audit/AUDIT.json'),required_arms=sorted(required),finished_training_arms=sorted(finished),
        completion_claim=required<=finished and len(evaluations)>=len(required),
        evaluation_count_alone_is_not_final_validation=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',default='/mnt/project/d44-native-rl-20261103');p.add_argument('--output');a=p.parse_args()
    result=summarize(a.root)
    if a.output:atomic_json(a.output,result)
    else:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
