"""Native scorer events and exact candidate replay; no factual future fiction."""
import torch
from starVLA.model.modules.qwen_world_compare.contracts import first_event_class
from starVLA.model.modules.qwen_world_compare.heads import evaluation_loss
from .safety_credit import FACTORS


def representative_indices(records, count=4):
    order=sorted(range(len(records)),key=lambda i:(records[i]['metrics']['score'],i))
    selected=[]
    for i in order[-2:][::-1]+order[:2]+order:
        if i not in selected:selected.append(i)
        if len(selected)==min(count,len(order)):break
    return selected


def labels(records, device):
    factors=torch.tensor([[r['metrics'][k]for k in FACTORS]for r in records],device=device)
    events=[]
    for r in records:
        road=r['non_drivable_body_by_time']
        indices=[i for i,v in enumerate(road)if bool(v)]
        firstroad=indices[0]*.1 if indices else None
        events.append([first_event_class(x)for x in (r['official_first_at_fault_collision_seconds'],
            firstroad,r['official_first_ttc_infraction_seconds'])])
    score=torch.tensor([r['metrics']['score']for r in records],device=device)
    event=torch.tensor(events,device=device,dtype=torch.long)
    return factors[None],torch.ones_like(factors[None],dtype=torch.bool),score[None],torch.ones_like(score[None],dtype=torch.bool),event[None],torch.ones_like(event[None],dtype=torch.bool)


def loss(model, condition, candidates, records, prefix):
    idx=representative_indices(records);x=candidates[idx].detach()
    mapping=torch.zeros(len(x),device=x.device,dtype=torch.long)
    r=model.replay_candidates(x,mapping,prefix)
    pred=model.readouts.evaluation(model.readouts.bridge(r),x)
    target=labels([records[i]for i in idx],x.device)
    result=evaluation_loss(pred,*target)
    result['event_accuracy']=(pred['event_logits'].argmax(-1)==target[4][0]).float().mean()
    return result
