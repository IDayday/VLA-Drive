"""Reuse strict fresh v2 predictions after verifying shared inference identity."""
from pathlib import Path
from common import ROOT, read, write, file_hash, poses_valid
from pipeline import verify


def main():
    cfg,new=verify();old_root=ROOT.parent/'mining_v2';old=read(old_root/'IDENTITY.json')
    for key in ('checkpoint_sha256','source_hashes','population_hash'):
        assert new[key]==old[key],key
    assert new['score_protocol']==old['score_protocol']
    old_cfg=read(old_root/'CONFIG.json')
    for key in ('checkpoint','source_root','tokenizer_root','seed','trajectory','sampling','failure'):
        assert cfg[key]==old_cfg[key],key
    receipt=read(old_root/'predictions/MODEL_LOAD.json')
    assert receipt['strict_load'] and receipt['identity']==old['identity']
    rows={row['token']:row for row in read(ROOT/'POPULATION.json')}
    destination=ROOT/'predictions';destination.mkdir(exist_ok=True);count=0
    for source in sorted((old_root/'predictions').glob('????????????????.json')):
        record=read(source);token=record['token']
        assert token==source.stem and token in rows and record['identity']==old['identity']
        assert record['checkpoint_sha256']==new['checkpoint_sha256']
        assert file_hash(rows[token]['raw_meta'])==record['raw_meta_sha256']
        poses_valid(record['trajectory'])
        record['identity']=new['identity']
        record['reuse_provenance']=dict(parent_identity=old['identity'],source=str(source),source_sha256=file_hash(source),
                                        strict_load_receipt_sha256=file_hash(old_root/'predictions/MODEL_LOAD.json'))
        target=destination/source.name
        if target.exists():assert read(target)==record
        else:write(target,record)
        count+=1
    write(destination/'MODEL_LOAD.json',dict(**{k:v for k,v in receipt.items() if k!='identity'},
          identity=new['identity'],reuse_parent_identity=old['identity']),replace=True)
    write(ROOT/'MIGRATION.json',dict(predictions=count,parent_identity=old['identity'],identity=new['identity'],
          policy='Prediction values unchanged; all GT labels re-scored/rebuilt under v3; v2 artifacts untouched'),replace=True)
    print(f'Validated and reused {count} fresh predictions',flush=True)


if __name__=='__main__':main()
