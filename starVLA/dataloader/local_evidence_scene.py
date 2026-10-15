"""Versioned metadata view; original strict scene/checker behaviour is unchanged."""
import json
from pathlib import Path
import numpy as np
from .minimal_repair_scene import load_annotated_scene


def load_local_evidence_scene(anchor, split, row):
    scene=load_annotated_scene(anchor,split,row)
    root=Path(anchor['data'][split]['label_root'])
    meta=json.loads((root/'records'/(row['token']+'.json')).read_text())
    quality=meta['height_quality'];base=quality[0].get('ground_level')
    flags=[]
    for q in quality:
        level=q.get('ground_level')
        flags.append(base is not None and level is not None and 'reason' not in q
                     and 'reason' not in quality[0] and abs(level-base)<=1.)
    scene.local_road_height_known=np.asarray(flags,bool)
    scene.map_identity='navsim_planar_drivable:'+row['log']
    scene.quality={**scene.quality,'local_evidence_height_rule':
        't0 and requested endpoints have valid ground height; each within 1m of t0; strict whole-plan flag unmodified'}
    return scene
