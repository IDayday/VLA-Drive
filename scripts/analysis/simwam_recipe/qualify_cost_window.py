"""Regression checks for unfinished attempts and resumed throughput accounting."""
import argparse
import json
from pathlib import Path
import tempfile

from scripts.analysis.simwam_recipe.cost_ledger import invocation_cost
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    checks=[]
    def calculate(rows,**kwargs):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for row in rows:
                dest=root/'invocations'/row['invocation'];dest.mkdir(parents=True,exist_ok=True)
                (dest/'rank0.json').write_text(json.dumps(row))
            return invocation_cost(root,**kwargs)
    current=dict(invocation='current',rank=0,status='RUNNING',start_unix=2000,
        start_update=1000,observed_worker_wall_seconds=140.,preparation_seconds=5.,
        seconds_save=2.,generated_candidates=400,scored_candidates=400)
    r=calculate([],done=1200,latest_wall=100)
    assert r['invocation_updates_per_second'] is None
    checks.append('legacy_no_fabricated_rate')
    r=calculate([current],done=1200,latest_wall=100)
    assert r['invocation_effective_updates'] is None and r['invocation_updates_per_second'] is None
    assert r['aborted_invocation_worker_wall_seconds']==0 and r['unclosed_invocation_worker_wall_seconds']==145
    checks.append('unclosed_not_aborted_or_assumed_live')
    r=calculate([current],done=1200,latest_wall=100,latest_invocation='current')
    assert r['invocation_effective_updates']==200 and r['invocation_updates_per_second']==2
    assert abs(r['linear_remaining_training_hours']-18800/2/3600)<1e-12
    checks.append('resumed_rate_uses_checkpoint_minus_start')
    r=calculate([current],done=1004,latest_wall=2,latest_invocation='current')
    assert r['invocation_effective_updates']==4 and r['invocation_updates_per_second'] is None
    checks.append('short_window_has_no_precise_ETA')
    previous=dict(current,invocation='previous',status='ABORTED',start_unix=1000,
                  start_update=0,end_update=1200,worker_wall_seconds=600.,preparation_seconds=10.)
    r=calculate([previous,current],done=1200,latest_wall=100,latest_invocation='current')
    assert r['cumulative_worker_wall_seconds']==755 and r['aborted_invocation_worker_wall_seconds']==610
    assert r['invocation_effective_updates']==200 and r['invocation_updates_per_second']==2
    checks.append('failed_attempt_and_loading_retained_without_double_logical_updates')
    terminal=dict(current,status='COMPLETE',start_update=19000,end_update=20000,
                  worker_wall_seconds=500.,preparation_seconds=10.)
    r=calculate([terminal],done=20000)
    assert r['invocation_effective_updates']==1000 and r['linear_remaining_training_hours']==0
    assert r['cumulative_worker_wall_seconds']==510
    checks.append('completed_segment_rate_and_full_elapsed')
    aborted=dict(current,status='ABORTED',end_update=1050,worker_wall_seconds=50.)
    r=calculate([aborted],done=1000)
    assert r['linear_remaining_training_hours'] is None and r['aborted_invocation_worker_wall_seconds']==55
    checks.append('aborted_attempt_not_a_completion_forecast')
    result=dict(status='PASS',checks=checks,scope='Analysis-only ledger schema3; no actor or inference changes')
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result);print(json.dumps(result))


if __name__=='__main__':main()
