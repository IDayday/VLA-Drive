"""Read actual progress/results; never substitute an old score for an unfinished arm."""
import argparse,json
from pathlib import Path
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--campaign-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.campaign_root);runs=[]
    for path in sorted((root/'students').glob('*/identity.json')):
        identity=json.loads(path.read_text());state=json.loads((path.parent/'status.json').read_text())
        runs.append({'run_id':path.parent.name,'source':identity['source_sha'],'identity':identity['identity'],
          'configuration':identity['config']['foresight'],'progress':state})
    evaluations=[{'file':str(path),'result':json.loads(path.read_text())} for path in sorted((root/'evaluations').glob('*_state.json'))]
    atomic_json(a.output,{'runs':runs,'development_evaluations':evaluations,'Navtest_policy':'final locked confirmation only; no new milestone observer'})

if __name__=='__main__':main()
