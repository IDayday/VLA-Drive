"""Seal an explicit user parent decision without altering native IL checkpoints."""
import argparse
import json
from pathlib import Path
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.qwen_sourceflow.loading import inspect_il_checkpoint,inspect_il
from starVLA.rl.qwen_sourceflow.selection import validate_user_authorization,validate_selection


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--authorization',required=True)
    p.add_argument('--observed-navtest-summary',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Parent selection is immutable')
    parent=inspect_il_checkpoint(Path(a.checkpoint).resolve())
    validate_user_authorization(a.authorization,parent)
    chosen=dict(checkpoint=parent['path'],updates=parent['seal']['updates'],
        model_sha256=parent['files']['model.pt'],training_identity=parent['identity']['sha256'])
    record=dict(schema='native_IL_user_selection_v1',selected=chosen,
        selected_model_sha256=chosen['model_sha256'],Navtest_used=True,
        selection_rule='explicit user-selected IL parent; observed Navtest disclosed',
        authorization_path=str(Path(a.authorization).resolve()),authorization_sha256=file_sha(a.authorization),
        observed_summary_path=str(Path(a.observed_navtest_summary).resolve()),
        observed_summary_sha256=file_sha(a.observed_navtest_summary))
    record['sha256']=fingerprint(record)
    temporary=output.with_suffix('.validating.json');atomic_json(temporary,record)
    validate_selection(temporary,parent)
    inspect_il(parent['path'],parent_selection=temporary)
    temporary.rename(output);print(json.dumps(record))


if __name__=='__main__':main()
