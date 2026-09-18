"""Register measured native pilot evidence for the explicit 64-update budget.

This file never writes READY or production acceptance. The native budget gate
reads the source artifacts and observed dtypes again at each actual launch.
"""
import argparse
import json
import os
from pathlib import Path
from scripts.cluster_flow_grpo.cluster import write_json
from starVLA.rl.flow_grpo.acceptance import acceptance_context, enforce_training_budget
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import file_sha


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('pilot','control','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();pilot=Path(args.pilot);output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    if not (pilot/'checkpoints/update_000002/COMPLETE').is_file():raise ValueError('native pilot not complete')
    def ref(path):
        path=Path(path).resolve();return {'path':str(path),'sha256':file_sha(path)}
    context=json.loads((pilot/'execution_context.json').read_text())
    rows=[json.loads(x) for x in (pilot/'training.jsonl').read_text().splitlines()]
    dtype=output.with_name('pilot_dtype.json');write_json(dtype,{'ranks':rows[-1]['ranks']})
    os.environ['WORLD_SIZE']='8'
    configs=[resolve_config(f'configs/flow_grpo/frozen_research_g16_{arm}64.yaml')[0] for arm in ('uniform','discount')]
    record={'schema_version':1,'status':'RESEARCH_ONLY','budget':64,
       'scope':'User requested longer F-only comparison; no production or historical chunk gate is closed.',
       'contexts':[acceptance_context(c,context['resume_identity']) for c in configs],
       'pilot_control':ref(args.control),'pilot_context':ref(pilot/'execution_context.json'),
       'pilot_config':ref(pilot/'rl_config.json'),'pilot_training':ref(pilot/'training.jsonl'),
       'pilot_dtype':ref(dtype),
       'pilot_immutable':[ref(pilot/f'immutable_update000002_rank{r}.json') for r in range(8)]}
    # The native reader expects a path. Publish the candidate under its final
    # research path, remove it if semantic validation rejects it. No trainer is
    # launched concurrently with this one-shot registration operation.
    write_json(output,record)
    try:
        for cfg,ctx in zip(configs,record['contexts']):enforce_training_budget(cfg,ctx)
    except BaseException:
        output.rename(output.with_name(output.name+'.rejected'))
        raise
    print(json.dumps({'status':'RESEARCH_ONLY','path':str(output),'contexts':record['contexts']},indent=2))


if __name__=='__main__':main()
