"""Summarize matched TRAIN sampler means; never turn finite best into deployment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha,fingerprint
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.flow_grpo.transactions import atomic_json


def summarize(root):
    root=Path(root);seal=json.loads((root/'COMPLETE.json').read_text())
    if file_sha(root/'summary_private.json')!=seal['summary_sha256']:raise ValueError('Changed probe summary')
    result=json.loads((root/'summary_private.json').read_text())
    rows=result['rows']; modes=('E_NATIVE','E_SHIFT5','SOURCE_SDE');stages=('SFT','RL')
    table={(r['stage'],r['mode'],r['token']):r for r in rows}
    tokens=sorted({r['token'] for r in rows}); logs=[table['SFT','E_NATIVE',t]['log'] for t in tokens]
    if len(tokens)!=64 or len(table)!=384 or len(rows)!=384:raise ValueError('Incomplete/duplicate probe population')
    values={};stats={};hashes=[];drift=[]
    for stage in stages:
        stats[stage]={}
        for mode in modes:
            selected=[table[stage,mode,t] for t in tokens]
            values[stage,mode]=np.array([r['mean']['score'] for r in selected])
            stats[stage][mode]=dict(mean={k:float(np.mean([r['mean'][k] for r in selected])) for k in selected[0]['mean']},
                safe_candidate_fraction=float(np.mean([r['safe_fraction'] for r in selected])),
                feasible_scene_fraction=float(np.mean([r['feasible'] for r in selected])),
                finite_best_safe_mean_conditional_on_feasible=float(np.mean([r['finite_best_safe'] for r in selected if r['feasible']])) if any(r['feasible'] for r in selected) else None,
                all_zero_groups=sum(r['all_zero'] for r in selected),any_zero_groups=sum(r['any_zero'] for r in selected))
    for token in tokens:
        pair=[]
        for stage in stages:
            path=root/stage/(token+'.pt'); meta=json.loads(path.with_suffix('.json').read_text())
            if file_sha(path)!=meta['payload_sha256']:raise ValueError('Changed actual probe trajectories')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if (payload['token']!=token or payload['stage']!=stage or tensor_digest(payload['root'])!=meta['root_sha256'] or
                meta['registration_sha256']!=result['identity']['registration_sha256']):raise ValueError('Unpaired probe identity')
            torch.testing.assert_close(payload['modes']['E_SHIFT5']['chain'][:,:8],
                payload['modes']['SOURCE_SDE']['chain'][:,:8],rtol=0,atol=0)
            for mode in modes:
                records=payload['modes'][mode]['records']
                if len(records)!=8 or any(r['status']!='OK' for r in records):raise ValueError('Invalid candidate denominator')
                score=np.mean([r['metrics']['score'] for r in records])
                if abs(score-table[stage,mode,token]['mean']['score'])>1e-12:raise ValueError('Unmatched saved scores')
            pair.append(payload);hashes.append(meta['payload_sha256'])
        if not torch.equal(pair[0]['root'],pair[1]['root']):raise ValueError('Unmatched roots across SFT/RL')
        drift.append((pair[1]['modes']['E_SHIFT5']['chain']-pair[0]['modes']['E_SHIFT5']['chain']).square().mean((0,2,3)).numpy())
    comparisons={}
    for mode in modes:
        comparisons[mode]=log_bootstrap_difference(values['RL',mode]-values['SFT',mode],logs)
    gaps={}
    for mode in ('E_NATIVE','E_SHIFT5'):
        gaps[mode]=dict(
            SFT_SDE_minus_ODE=log_bootstrap_difference(values['SFT','SOURCE_SDE']-values['SFT',mode],logs),
            RL_SDE_minus_ODE=log_bootstrap_difference(values['RL','SOURCE_SDE']-values['RL',mode],logs),
            SDE_RL_gain_minus_ODE_RL_gain=log_bootstrap_difference(
                values['RL','SOURCE_SDE']-values['SFT','SOURCE_SDE']-values['RL',mode]+values['SFT',mode],logs))
    return dict(status='COMPLETE_MATCHED_SAMPLER_ANALYSIS',scenes=64,logs=len(set(logs)),G=8,
        identity=result['identity'],payload_digest=fingerprint(hashes),modes=stats,RL_minus_SFT=comparisons,
        sampler_gaps=gaps,shift5_latent_RL_minus_SFT_RMS_per_generation_state=np.sqrt(np.mean(drift,axis=0)).tolist(),
        exact_common_root_and_first_seven_ODE_steps=True,
        cost={k:result[k] for k in ('new_candidates','actual_scoring_calls','wall_seconds','optimizer_updates')},
        limits=result['limits']+['Latent generation drift is not physical-time risk or causal prefix attribution.',
            'Confidence intervals aggregate candidates per scene before log clustering; not scene-times-seed independent samples.'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--probe',required=True);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();require_dispatch()
    if a.output.exists():raise FileExistsError(a.output)
    result=summarize(a.probe);result['analysis_sha256']=file_sha(__file__)
    atomic_json(a.output,result);print(json.dumps(dict(status=result['status'],RL_minus_SFT=result['RL_minus_SFT'])))


if __name__=='__main__':main()
