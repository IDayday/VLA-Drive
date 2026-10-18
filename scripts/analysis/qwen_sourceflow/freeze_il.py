"""Select once from all five complete DEV milestones, then seal the IL parent."""
import argparse
import json
from pathlib import Path
from starVLA.rl.qwen_sourceflow.contracts import file_sha,fingerprint
from starVLA.rl.qwen_sourceflow.loading import inspect_il
from starVLA.rl.qwen_sourceflow.selection import validate_selection
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--summaries',nargs=5,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('IL parent is immutable')
    candidates=[]
    for path in a.summaries:
        summary=json.loads(Path(path).read_text());parent=inspect_il(summary['checkpoint'])
        candidates.append(dict(updates=parent['seal']['updates'],summary_path=str(Path(path).resolve()),
            summary_sha256=file_sha(path),PDMS=summary['means']['score'],checkpoint=parent['path'],
            model_sha256=parent['files']['model.pt'],training_identity=parent['identity']['sha256']))
    if len({x['training_identity'] for x in candidates})!=1:raise ValueError('Mixed IL runs')
    # Tie rule: prefer the later registered checkpoint, set before seeing scores.
    chosen=max(candidates,key=lambda x:(x['PDMS'],x['updates']))
    record=dict(schema='native_IL_DEV_selection_v1',candidates=candidates,selected=chosen,
        selected_model_sha256=chosen['model_sha256'],selection_rule='highest DEV PDMS; exact tie chooses later update',
        Navtest_used=False)
    record['sha256']=fingerprint(record)
    # Validate before publishing, including all five DEV population/protocols.
    temp=Path(a.output).with_suffix('.validating.json');atomic_json(temp,record)
    validate_selection(temp,inspect_il(chosen['checkpoint']))
    temp.rename(a.output);print(json.dumps(chosen))


if __name__=='__main__':main()
