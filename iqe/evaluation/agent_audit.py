"""Real NAVSIM AgentInput versus the source framework's image/ego preprocessing."""
from pathlib import Path
import pickle
import numpy as np
from ..agent import IQEAgent, observation_from_agent_input
from ..contracts import require
from ..data.sources import load_observation
from ..io import read_json,file_hash


def real_agent_input(scene, contract, raw_log_root):
    from navsim.common.dataclasses import AgentInput
    log=Path(raw_log_root)/(scene.source_log_id+'.pkl')
    with log.open('rb') as f:frames=pickle.load(f)
    matches=[i for i,frame in enumerate(frames) if frame['token']==scene.scene_id]
    require(len(matches)==1 and matches[0]>=3,'real Agent scene/history join')
    history=frames[matches[0]-3:matches[0]+1]
    record=read_json(scene.observation_ref)
    image=Path(record['image_paths'][0]);relative=Path(history[-1]['cams']['CAM_F0']['data_path'])
    require(image.parts[-len(relative.parts):]==relative.parts,'raw log / observation image mismatch')
    sensor_root=image.parents[len(relative.parts)-1]
    config=IQEAgent('audit_only_not_loaded',device='cpu',allow_smoke=True).get_sensor_config()
    data=AgentInput.from_scene_dict_list(history,sensor_root,4,config)
    original=load_observation(scene,contract);converted=observation_from_agent_input(data)
    require(original['lang']==converted['lang'],'Agent prompt differs from original source')
    require(all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(original['image'],converted['image'])),
            'Agent image preprocessing differs from original source')
    error=float(np.abs(original['state']-converted['state']).max())
    require(error<=1e-6,'Agent ego transform differs from source beyond float32 NAVSIM pose tolerance')
    return data,{'scene_id':scene.scene_id,'raw_log_hash':file_hash(log),'real_NAVSIM_AgentInput':True,
                 'image_pixels_equal':True,'prompt_equal':True,'ego_state_max_abs':error,
                 'input_frames':4,'future_input_frames':0}
