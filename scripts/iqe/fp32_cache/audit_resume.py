"""Verify consumed-only continuation and attach actual next IDs to its receipt."""
import argparse
from pathlib import Path
from iqe.io import atomic_json, read_json, file_hash
from iqe.contracts import require
from iqe.performance.compare import compare_resume
from iqe.training.resume import record_next_batch


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--resumed',type=Path,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=compare_resume(a.reference,a.resumed)
    result=read_json(a.resumed/'RESULT.json')
    require(result['initial_step']==33001 and result['final_step']==33002,'real first-two-batch resume diagnostic')
    mine=[read_json(a.resumed/f'rank_{r}.json')['rows'][0]['sample_ids'] for r in range(8)]
    ids=[mine[rank][offset] for offset in range(4) for rank in range(8)]
    record_next_batch(a.resumed,ids,exact=report['status']=='PASS')
    report.update(actual_restored_next_sample_ids=ids,optimizer_steps_added=1,
        checkpoint=str(a.resumed/'step_033002.pt'),checkpoint_hash=file_hash(a.resumed/'step_033002.pt'),
        scope='all eight ranks: full model/Adam tensor fingerprints, loss/gradient norm/LR, all RNG and consumed cursor',
        previous_complete_checkpoint_retained=(a.resumed/'step_033001.pt.COMPLETE.json').exists(),
        historical_pause_result_retained=(a.resumed/'history/step_033001/RESULT.json').exists())
    atomic_json(a.output,report);print({'status':report['status'],'report':a.output})
    return 0 if report['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())
