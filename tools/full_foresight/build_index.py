"""Version the verified physical-time index in training-scene encounter order.

Image keys and source identities are unchanged. Only storage offsets change;
this permits a bounded prefix to cover all four times of early training scenes.
"""
import argparse
import json
from pathlib import Path
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256


def build(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('New immutable index required')
    ident = json.loads((source/'identity.json').read_text())
    if ident['schema'] != 'dinov3_image_index_v1' or ident['horizons_s'] != [0.,1.,2.,4.]:
        raise ValueError('Expected verified current/future physical-time source')
    if identity_hash({k:v for k,v in ident.items() if k != 'identity'}) != ident['identity']:
        raise ValueError('Source identity modified')
    for name, fname in [('images','images.json'),('train','train_scenes.json'),('dev','dev_scenes.json')]:
        if file_sha256(source/fname) != ident['files'][name]:
            raise ValueError('Source index bytes changed')
    old = json.loads((source/'images.json').read_text())
    rows = {s:json.loads((source/(s+'_scenes.json')).read_text()) for s in ('train','dev')}
    if {r['log'] for r in rows['train']} & {r['log'] for r in rows['dev']}:
        raise ValueError('Train/dev log leakage')
    remap, images = {}, []
    for split in ('train','dev'):
        for row in rows[split]:
            if row['requested_horizons_s'] != [0.,1.,2.,4.] or row['actual_horizons_s'][0] != 0:
                raise ValueError('Current decision/time meaning changed')
            for k, requested in enumerate(row['requested_horizons_s']):
                actual = row['actual_horizons_s'][k]
                if actual is not None and abs(actual-requested)>ident['timestamp_tolerance_s']+1e-9:
                    raise ValueError('Timestamp tolerance violated')
                for view, old_id in enumerate(row['images'][k]):
                    if old_id < 0:
                        if k == 0:raise ValueError('Missing current image')
                        continue
                    item = old[old_id]
                    if item['split'] != split or item['log'] != row['log'] or item['view'] != view:
                        raise ValueError('Cross-split/log/view image reference')
                    if abs((item['timestamp']-row['timestamp'])/1e6-requested)>ident['timestamp_tolerance_s']+1e-9:
                        raise ValueError('Source image physical time mismatch')
                    if old_id not in remap:
                        remap[old_id] = len(images); images.append(item)
                    row['images'][k][view] = remap[old_id]
    output.mkdir(parents=True)
    atomic_json(output/'images.json', images)
    files = {'images':file_sha256(output/'images.json')}
    for split, data in rows.items():
        atomic_json(output/(split+'_scenes.json'), data)
        files[split] = file_sha256(output/(split+'_scenes.json'))
    new = {k:ident[k] for k in ('partition_sha256','index_hashes','timestamp_tolerance_s','horizons_s','views')}
    new.update(schema='ddp_full_dino_index_v1', files=files, source_index=ident['identity'],
               source_identity_sha256=file_sha256(source/'identity.json'),
               writer_sha256=file_sha256(__file__), storage_order='first occurrence in train then dev; each scene h0/1/2/4, view F/L/R')
    new['identity'] = identity_hash(new); atomic_json(output/'identity.json', new)
    atomic_json(output/'COMPLETE.json', {'identity':new['identity'],'images':len(images),
        'scenes':{s:len(r) for s,r in rows.items()},'logs':{s:len({x['log'] for x in r}) for s,r in rows.items()},
        'bytes_six_fp16_targets':len(images)*636*1024*2, 'source_rgb_bytes':sum(x['bytes'] for x in images),
        'train_dev_log_overlap':0})
    return new


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-index',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(build(a.source_index,a.output)['identity'])

if __name__=='__main__':main()
