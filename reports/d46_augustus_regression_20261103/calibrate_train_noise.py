"""All103288 TRAIN GT quantiles, no images, scorer, CUDA or NAVTEST labels."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from starVLA.rl.d46_augustus.config import defaults
from starVLA.rl.d44_native_rl.data import DataStream
from starVLA.rl.d46_augustus.physical_noise import calibrated_scale,codec_jacobian
from starVLA.rl.qwen_world_compare.contracts import fingerprint,file_sha

def main():
 start=time.monotonic();stream=DataStream(defaults('AUG2_LORA'),128)
 target=np.asarray([stream.data.records[t]['requested_GT']for t in stream.tokens],dtype=np.float64)
 assert target.shape==(103288,8,3)and np.isfinite(target).all()
 low,high=np.quantile(target,[.01,.99],axis=0);scale,physical=calibrated_scale(low,high)
 old=codec_jacobian()*np.array([.05,.05,.02])
 result=dict(schema='d46_full_TRAIN_physical_exploration_calibration_v1',status='CPU_CALIBRATED_NOT_IN_RUNNING_RECIPES',
  count=len(target),data=stream.identity,actor_codec_unchanged=True,NAVTEST_used=False,statistics_only_not_actor_inputs=True,
  reference_source='pinned upstream model_qwen3p5 normalize_actions uses per-timestep q01/q99',
  q01=low.tolist(),q99=high.tolist(),relative_scale=[.05,.05,.02],physical_floor_multiplier=[.02,.02,.002],
  D_codec_jacobian=codec_jacobian().tolist(),original_fixed_physical_multiplier=old.tolist(),
  calibrated_D_noise_scale=scale.tolist(),calibrated_physical_multiplier=physical.tolist(),
  last_native_transition_std_factor=.3571428656578064,
  limits='Physical coordinate conversion, not evidence of higher PDMS. Floors are a registered proposal; require real GPU gradient/ratio and B1 assessment. No current training source changed.',
  seconds=time.monotonic()-start,source_file_sha256=file_sha(__file__))
 result['sha256']=fingerprint(result);out=Path('/mnt/project/d46-priority-rl-20261103/analysis/AUGUSTUS_REGRESSION_20261103/TRAIN103K_NOISE_CALIBRATION.json');out.write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(status=result['status'],first_physical=physical[0].tolist(),last_physical=physical[-1].tolist(),old_physical=old[0].tolist(),seconds=result['seconds'])))
if __name__=='__main__':main()
