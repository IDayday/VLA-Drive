"""A-only index view over deterministic GT records, never a fitted full-data P.

The source record identity is preserved, while the role/index identity is
separate and explicit. The original teacher trainer saves both selected index
hashes, optimizers, masks, RNG, and data offset. No original file is modified.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest,identity,check_host,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-data','role-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();check_host()
    origin=read(a.source_data/'identity.json');audit=read(a.role_root/'SOURCE_AUDIT.json')
    if origin['schema']!='foresight_gt_vehicle_mae_v1':raise ValueError('Wrong deterministic teacher input schema')
    parent={r['token']:r for r in read(a.source_data/'train_index.json')}
    train=read(a.role_root/'A_train.json');cal=read(a.role_root/'A_calibration.json')
    for rows in (train,cal):
        if any(parent.get(r['token'])!=r for r in rows):raise ValueError('A-only rows not from legal parent training data')
    if {r['log'] for r in train}&{r['log'] for r in cal}:raise ValueError('Teacher calibration overlaps fitting')
    contract=dict(schema='a_only_deterministic_teacher_input_view_v1',source_data=str(a.source_data),
        source_record_identity=origin['identity'],source_identity_sha256=digest(a.source_data/'identity.json'),
        population_identity=audit['identity'],train_index_identity=identity(train),calibration_index_identity=identity(cal),
        training_role='A_train',calibration_role='A_calibration',source_sha=source_sha(ROOT),
        fitting_init='random teacher; no full-data teacher checkpoint permitted',
        source_scope='Deterministic raw trajectories/graph inputs, not fitted teacher features; source record identity retained',
        selection='Original30epoch teacher recipe; choose fixed registered milestones using A-only calibration')
    contract['identity']=identity(contract)
    if a.output.exists():raise FileExistsError('Do not overwrite a teacher role view')
    a.output.mkdir(parents=True)
    (a.output/'records').symlink_to((a.source_data/'records').resolve(),target_is_directory=True)
    atomic_json(a.output/'identity.json',origin)
    atomic_json(a.output/'train_index.json',train);atomic_json(a.output/'dev_index.json',cal)
    atomic_json(a.output/'VIEW.json',contract)
    atomic_json(a.output/'COMPLETE.json',dict(identity=contract['identity'],train_scenes=len(train),calibration_scenes=len(cal)))
    print(contract['identity'])


if __name__=='__main__':main()
