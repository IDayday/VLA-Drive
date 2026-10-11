"""Real FP32 payload regrouping audit; never substitutes for update/resume proof."""
import argparse
from pathlib import Path
import json
import torch
from iqe.config import load_config
from iqe.io import atomic_json, read_json, digest
from iqe.resources import qualify
from iqe.vision_cache.cli import load_front, hold_check
from iqe.vision_cache.plan import load_plan
from iqe.vision_cache.builder import FrozenFront
from iqe.vision_cache.identity import visual_identity
from iqe.vision_cache.inputs import current_numeric_fields
from iqe.vision_cache.storage import CacheReader, encode_record, decode_record


def tensors(p):
    return {'ids':p.input_ids,'positions':p.position_ids,'grid':p.image_grid_thw,'image':p.image_embeddings,
            **{f'deepstack_{i}':t for i,t in enumerate(p.deepstack)},**{f'slot_{n}':t for n,t in p.slots.items()}}


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('config','plan','root','output','device','hold-file'):p.add_argument('--'+n,required=True)
    a=p.parse_args();hold_check(a.hold_file);qualify(a.device);torch.set_num_threads(2)
    plan=load_plan(a.plan);model,contract=load_front(load_config(a.config),plan,a.device)
    reader=CacheReader(a.root,identity=visual_identity(model,contract))
    from starVLA.dataloader.foresight_dataset import current_observation
    rows=plan['rows'][:8]; observations=[current_observation(read_json(r['current_ref'])) for r in rows]
    records=[];exact=True
    for row,obs in zip(rows,observations):
        state,lang=current_numeric_fields(read_json(row['current_ref']))
        assert (state==obs['state']).all() and lang==obs['lang']
    front=FrozenFront(model)
    cases={'original_four':[0,1,2,3],'batch_one':[0],'shuffled_four':[3,1,0,2],
           'duplicate_four':[0,0,1,1],'tail_two':[0,3],'next_four':[4,5,6,7]}
    for case,indices in cases.items():
        online=front([observations[i] for i in indices])
        for index,actual in zip(indices,online):
            cached=reader.read(rows[index]['scene_id']);roundtrip=decode_record(encode_record(cached))
            row={'case':case,'scene':rows[index]['scene_id'],'tensors':{}}
            for name,t in tensors(actual).items():
                v=tensors(cached)[name];s=tensors(roundtrip)[name]
                equal=torch.equal(t,v);exact &= equal
                row['tensors'][name]={'shape_equal':t.shape==v.shape,'dtype_equal':t.dtype==v.dtype,
                    'bitwise':equal,'storage_bitwise':torch.equal(s,v),
                    'max_abs':float((t.double()-v.double()).abs().max()) if t.numel() else 0.}
            records.append(row)
    front.close();reader.close()
    result={'status':'PASS' if exact else 'FAIL_FP32_BATCH_VARIATION','rows':records,
        'scope':'real image, frozen visual and every payload element; no trainable update comparison',
        'policy':'bitwise original and regrouped payloads; any deviation is reported, no BF16 tolerance reused',
        'storage_bitwise':all(v['storage_bitwise'] for r in records for v in r['tensors'].values()),
        'all_payloads_bitwise_equal':exact,'deepstack_count':3,'sample_sequence_sha256':digest(plan['sample_sequence'])}
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    return 0 if exact else 2

if __name__=='__main__':raise SystemExit(main())
