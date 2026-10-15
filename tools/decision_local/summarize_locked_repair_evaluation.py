"""Aggregate locked models by run ID, keeping different MIN recipes separate."""
import argparse,csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest
from tools.decision_local.summarize_minimal_repair_results import evaluator_contract
from tools.local_interaction_mask_v2.compare_pdms import read as read_scores,compare,METRICS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reg=read(a.registry);tables=[];banks={};paired={};protocol=None;input_bank=None
    expected={'dev':1696,'navtest':12146}[reg['population']]
    for job in reg['jobs']:
        name=job['id'];root=Path(job['scores']);done=read(root/'COMPLETE.json');meta=read(root/'identity.json')
        pred=read(Path(job['predictions'])/'identity.json');train=read(Path(job['checkpoint']).parent/'identity.json')
        if name in banks or job['epoch']!=24 or train['kind']!=job['kind'] or train['seed']!=job['seed']:
            raise ValueError('Duplicate ID or foreign common endpoint')
        if meta['prediction_identity']!=pred['identity'] or pred['checkpoint_sha256']!=digest(job['checkpoint']):
            raise ValueError('Different prediction/checkpoint identity')
        current=evaluator_contract(meta)
        if protocol is not None and current!=protocol:raise ValueError('Mixed canonical protocols')
        protocol=current
        if any(not done[v]['valid'] or done[v]['failed'] or done[v]['scenes']!=expected for v in ('q0','q_final')):
            raise ValueError('Incomplete or failed formal population')
        orig,final=[read_scores(root/v/'scenes.csv') for v in ('q0','q_final')]
        if set(orig)!=set(final) or len(orig)!=expected:raise ValueError('Different scene sets')
        if input_bank is not None:
            if set(orig)!=set(input_bank) or any(orig[t].get(k)!=input_bank[t].get(k) for t in orig for k in METRICS):
                raise ValueError('Changed initial-plan scores')
        input_bank=orig;banks[name]=final;comparison,transitions=compare(final,orig);paired[name+' minus input']=comparison
        row={'run_id':name,'label':job.get('analysis_label',name),'kind':job['kind'],'seed':job['seed'],
            'epoch':24,'scenes':expected,'training_source_sha':train['source_sha'],'information_mode':pred['mode'],
            'PDMS':done['q_final']['metrics']['score']*100,'input_PDMS':done['q0']['metrics']['score']*100,
            'delta_input_pp':comparison['delta_percentage_points'],
            'zero_to_nonzero':sum(x['baseline_score']==0 and x['first_score']>0 for x in transitions),
            'nonzero_to_zero':sum(x['baseline_score']>0 and x['first_score']==0 for x in transitions),
            **{k:done['q_final']['metrics'][k] for k in METRICS if k!='score'}}
        tables.append(row)
    for first,second in reg.get('comparison_pairs',[]):
        if first not in banks or second not in banks:raise ValueError('Unregistered comparison')
        paired[first+' minus '+second]=compare(banks[first],banks[second])[0]
    a.output.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output/'SUMMARY.json',{'registry':str(a.registry),'registry_sha256':digest(a.registry),'protocol':protocol,
        'population':reg['population'],'tables':tables,'paired':paired,'tool_sha256':digest(__file__),
        'scope':'Separate run IDs preserve original and bounded-follow-up scientific identities; common24, no score-based output selection',
        'limitations':'Fixed shared S0 seed; head seeds are not full VLA replicates; previously observed Navtest is not a new blind set'})
    with (a.output/'RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,list(tables[0]));writer.writeheader();writer.writerows(tables)
    print({x['label']:round(x['PDMS'],5) for x in tables})


if __name__=='__main__':main()
