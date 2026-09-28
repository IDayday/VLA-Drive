"""Freeze completed, dev-selected formal models before any Navtest export."""
import argparse
import json
from pathlib import Path
import subprocess
from .checkpoints import checkpoint_identity
from .prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--selection',required=True,help='JSON list of run, tag, and complete dev scoring directories for the fixed seed list')
    p.add_argument('--output',required=True)
    p.add_argument('--sampling-seeds',nargs='+',type=int,default=[42,43,44,45,46])
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('A final-model lock is immutable')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain']):raise ValueError('Commit evaluation implementation first')
    selected=json.loads(Path(a.selection).read_text());models=[];seen=set()
    for entry in selected:
        run, ckpt=checkpoint_identity(entry['run'],entry['tag'])
        key=(run['arm'],int(run['config']['seed']))
        if key in seen or run['startup']:raise ValueError('Duplicate or startup model cannot enter final table')
        seen.add(key)
        status=json.loads((Path(entry['run'])/'status.json').read_text())
        if status['status']!='COMPLETE' or status['real_optimizer_updates']!=100000:
            raise ValueError('Final original-recipe training is incomplete')
        if len(entry['dev_scores'])!=len(a.sampling_seeds):raise ValueError('Dev inference seed protocol incomplete')
        dev=[]
        for folder,seed in zip(entry['dev_scores'],a.sampling_seeds):
            folder=Path(folder);summary=json.loads((folder/'summary.json').read_text())
            scoring=json.loads((folder/'identity.json').read_text())
            bank=json.loads((Path(scoring['arguments']['predictions'])/'identity.json').read_text())
            if not summary['valid'] or summary['scenes']!=1696 or summary['logs']!=16 or summary['failed']:
                raise ValueError('A selected model needs the complete fixed development population')
            if bank['checkpoint']['sha256']!=ckpt['sha256'] or bank['protocol']['sampling_seed']!=seed:
                raise ValueError('Dev evidence belongs to another checkpoint or sampling seed')
            dev.append({'summary_sha256':file_sha256(folder/'summary.json'),'scenes_sha256':file_sha256(folder/'scenes.csv'),'sampling_seed':seed})
        models.append({'arm':key[0],'training_seed':key[1],'checkpoint':ckpt,'dev':dev})
    if not {('A',42),('B',42),('C',42),('B',43),('C',43)}.issubset(seen):
        raise ValueError('Primary A/B/C and the requested B/C second training seed are incomplete')
    atomic_json(a.output,{'evaluation_source_sha':source,'sampling_seeds':a.sampling_seeds,'inference_steps':10,
                         'precision':'FP32','candidate_count':1,'scorer':None,
                         'checkpoint_sha256':[m['checkpoint']['sha256'] for m in models],'models':models,
                         'selection_sha256':file_sha256(a.selection),'navtest_results_used_for_selection':False})


if __name__=='__main__':main()
