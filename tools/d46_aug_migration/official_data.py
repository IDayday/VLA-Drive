"""Small TRAIN-only AUG input reconstruction, using its original data helpers.

This is an action-only diagnostic: no invented CoT annotations. History is
reconstructed from NAVSIM ego poses, not the separately distributed P3 labels.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from moviepy.editor import ImageSequenceClip

from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=8);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    upstream=Path(a.upstream);dp=upstream/'AUGUstus_SFT/data_process';sys.path.insert(0,str(dp))
    spec=importlib.util.spec_from_file_location('aug_format',dp/'4_1_process_json.py')
    formatting=importlib.util.module_from_spec(spec);spec.loader.exec_module(formatting)
    data=WorldSceneData('/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN',role='NAVTRAIN')
    pop=json.loads(Path(data.manifest['source_population_path']).read_text())
    test=set(pop['navtest_tokens']);tokens=data.tokens[:a.count]
    assert not set(tokens)&test
    raw=Path('/mnt/project/DriveDreamer-Policy/navsim_raw');samples=[]
    prompt=(upstream/'AUGUstus_SFT/prompts/default.txt').read_text()
    for token in tokens:
        record=data.records[token];log=record['log']
        frames=pickle.load((raw/'navsim_logs/trainval'/f'{log}.pkl').open('rb'))
        i=next(i for i,r in enumerate(frames)if r['token']==token)
        assert i>=3
        history=frames[i-3:i+1];current=history[-1]
        yaw=Quaternion(current['ego2global_rotation']).yaw_pitch_roll[0]
        origin=np.asarray(current['ego2global_translation'])[:2]
        rotation=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
        poses=[];images=[];folder=out/token;folder.mkdir(exist_ok=True)
        for j,row in enumerate(history):
            xy=rotation@(np.asarray(row['ego2global_translation'])[:2]-origin)
            angle=Quaternion(row['ego2global_rotation']).yaw_pitch_roll[0]-yaw
            angle=np.arctan2(np.sin(angle),np.cos(angle))
            poses.append([float(xy[0]),float(xy[1]),float(angle)])
            cams=[cv2.imread(str(raw/'sensor_blobs/trainval'/row['cams'][c]['data_path']))for c in ('CAM_L0','CAM_F0','CAM_R0')]
            if any(x is None for x in cams):raise FileNotFoundError('Missing real history image')
            left,front,right=cams
            stitched=np.concatenate([left[28:-28,416:-416],front[:-56],right[28:-28,416:-416]],axis=1)
            target=folder/f'{j}.jpg';assert cv2.imwrite(str(target),cv2.resize(stitched,(2048,512)))
            with Image.open(target)as im:images.append(np.array(im.convert('RGB')))
        # Same JPEG -> libx264 video path as official stages 2_0 / 2_1.
        video=folder/'history.mp4'
        clip=ImageSequenceClip(images,fps=2.)
        clip.write_videofile(str(video),codec='libx264',audio=False,logger=None,threads=2);clip.close()
        fields=dict(driving_command=formatting.decode_driving_command(current['driving_command']),
                    ego_dynamic_state=formatting.decode_ego_dynamic_state(current['ego_dynamic_state']),
                    historical_x_y_angle=str(poses))
        samples.append(dict(id=token,token=token,log=log,text=prompt.format(**fields),
                            media_path=str(video),media_type='video',actions=record['requested_GT'],
                            historical_poses=poses,history_timestamps=[int(r['timestamp'])for r in history]))
    atomic_json(out/'samples.json',samples)
    atomic_json(out/'MANIFEST.json',dict(status='COMPLETE',count=len(samples),role='NAVTRAIN',
        NAVTEST_overlap=0,source_manifest_sha=data.sha,source=str(upstream),
        input_protocol='4 real history frames, 2Hz, official crop/stitch/JPEG/libx264, 2048x512',
        difference='History reconstructed from NAVSIM poses; P3 JSON rounding not asserted identical. No CoT targets.',
        canonical_metric_cache_roots=data.manifest['canonical_metric_cache_roots']))


if __name__=='__main__':main()
