"""Use qualified byte prefetch for the unchanged complete comparison."""
import argparse
from pathlib import Path
import time

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from .analyze_cached_scores import cached_score_bytes
from .lock_final import read
from .summarize_interim_navtest import summarize


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('root','population','output','acceptance'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();require_dispatch();root=Path(a.root)
    proof=read(a.acceptance)
    adapter=Path(__file__).with_name('analyze_cached_scores.py')
    qualified=read(Path(a.acceptance).with_name('report.json'))
    if (not proof.get('exact_original_report_match') or proof['source_sha256']!=file_sha(adapter)
            or qualified['scenes']!=12146 or qualified['logs']!=136
            or qualified['sha256']!=proof['report_sha256']):
        raise ValueError('Require full-population original-analyzer parity for this adapter')
    while not all((root/m/'cost.json').is_file() and (root/m/'report.json').is_file()
                  for m in ('SFT','C_SDE','N_FLOW_NOISE')):
        require_dispatch()
        failures=[x for m in ('SFT','C_SDE','N_FLOW_NOISE') for x in (root/m).glob('failure_*.json')]
        if failures:
            raise RuntimeError('Original evaluation failed; preserve evidence: '+str(failures))
        time.sleep(10)
    population=read(a.population);started=time.monotonic()
    with cached_score_bytes([root/m/'scores' for m in ('SFT','C_SDE','N_FLOW_NOISE')],population) as evidence:
        result=summarize(root,population,a.output)
    evidence.update(wall_seconds=time.monotonic()-started,original_summary_unchanged=True,
                    comparison_sha256=result['sha256'],acceptance_sha256=file_sha(a.acceptance),
                    adapter_source_sha256=file_sha(adapter),source_sha256=file_sha(__file__))
    atomic_json(root/'comparison_analysis_execution.json',evidence)
    print({m:100*x['metrics']['score']['current'] for m,x in result['models'].items()},flush=True)


if __name__=='__main__':main()
