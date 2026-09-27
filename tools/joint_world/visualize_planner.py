"""Join current detections and generated joint/DiT trajectories for diagnostic figures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from tools.joint_world.planner_runtime import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['predictions','cache','targets','sensor-root','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--limit',type=int,default=32)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);joined=out/'joined_predictions';joined.mkdir()
    manifest=json.loads((Path(a.cache)/'manifest.json').read_text());records={r['token']:r for r in manifest['records']}
    for path in sorted(Path(a.predictions).glob('*.npz')):
        token=path.stem;source=Path(a.cache)/(token+'.pt')
        if file_sha256(source)!=records[token]['sha256']:raise ValueError('Changed current features')
        current=torch.load(source,map_location='cpu',weights_only=True)
        with np.load(path) as arrays:
            np.savez(joined/path.name,boxes=current['current_boxes'][0].numpy(),logits=current['current_logits'][0].numpy(),
                     future_xy=arrays['joint_xy'][1:],ego_trajectory=arrays['trajectory'],joint_ego_xy=arrays['joint_xy'][0])
    subprocess.run([sys.executable,'-m','tools.structured_world.visualize','--predictions',str(joined),
                    '--target-cache',a.targets,'--sensor-root',a.sensor_root,'--output',str(out/'figures'),
                    '--limit',str(a.limit),'--matching','geometry'],check=True)
    rows=json.loads((out/'figures/index.json').read_text())
    for row in rows:row['sha256']=hashlib.sha256(Path(row['file']).read_bytes()).hexdigest()
    (out/'VISUALIZATION_INDEX.json').write_text(json.dumps({'records':rows,'figure_selection':'Post-inference diagnostic categories, never model input selection',
        'private_images':'Figures remain in the private artifact directory; only this index is published.'},indent=2)+'\n')


if __name__=='__main__':main()
