"""Run the exact S0 trainer with one explicit, audited A-only data adapter.

No optimizer/model/sampler is reimplemented. The dependency injection replaces
only that process's dataset constructor. Historical files/site-packages remain
read-only. The run config/registration must bind this wrapper and legacy SHA.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-source',required=True,type=Path)
    p.add_argument('--population-contract',required=True,type=Path)
    p.add_argument('trainer_args',nargs=argparse.REMAINDER);a=p.parse_args()
    helper=load_file('_a_only_helpers',ROOT/'tools/decision_local/common.py');resource=helper.check_host()
    if helper.source_sha(a.legacy_source)!='1493deda247107efe076b9294863a6753391298a':
        raise ValueError('Require the real original S0 training implementation')
    contract=helper.read(a.population_contract)
    proof=helper.read(Path(contract['recording_frame_audit']))
    if not proof['population_ready'] or proof['source_audit_identity']!=contract['source_audit_identity']:
        raise ValueError('Recording/duplicate-image isolation is not complete')
    if contract['execution_wrapper_sha256']!=helper.digest(__file__):raise ValueError('Base execution wrapper changed')
    adapter=ROOT/'starVLA/dataloader/a_only_base_view.py'
    if contract['dataset_adapter_sha256']!=helper.digest(adapter):raise ValueError('A-only data adapter changed')
    # The legacy runtime is imported first, just like frozen S0 cache/export.
    sys.path.insert(0,str(a.legacy_source))
    from starVLA.dataloader import action_video_foresight_dataset as data_module
    original=data_module.ActionVideoForesightDataset
    view=load_file('_a_only_dataset',adapter).AOnlyBaseView
    def scoped_dataset(*args,**kwargs):
        return view(original,*args,population_contract=a.population_contract,**kwargs)
    data_module.ActionVideoForesightDataset=scoped_dataset
    from tools.foresight import train_student
    args=a.trainer_args[1:] if a.trainer_args[:1]==['--'] else a.trainer_args
    if not args:raise ValueError('Explicit finite registered trainer arguments required')
    for option in ('--campaign-gpu-hours','--max-seconds','--registration'):
        if option not in args:raise ValueError('Finite budget and role registration required: '+option)
    config=helper.read(Path(args[args.index('--config')+1]))
    ancestry=config.get('natural_error_provenance',{})
    if ancestry.get('population_contract_sha256')!=helper.digest(a.population_contract):
        raise ValueError('Checkpoint config must bind complete base provenance')
    if ancestry.get('legacy_training_source_sha')!=helper.source_sha(a.legacy_source):raise ValueError('Unbound legacy training source')
    sys.argv=[str(a.legacy_source/'tools/foresight/train_student.py'),*args]
    train_student.main()


if __name__=='__main__':main()
