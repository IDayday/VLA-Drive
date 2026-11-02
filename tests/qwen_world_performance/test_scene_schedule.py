import copy
import torch
from starVLA.rl.qwen_world_performance.stage_scheduler import SceneExposureScheduler, lr_multiplier

SPEC=dict(warmup_epochs=.08,hold_until_epoch=.75,cosine_decay_end_epoch=4.,minimum_lr_ratio=.05)
def test_clock_ignores_inner_reuse_and_roundtrips():
 p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([p],lr=5e-5)
 s=SceneExposureScheduler(o,SPEC,103288)
 s.set_exposures(128);expected=o.param_groups[0]['lr']
 for _ in range(4):s.step()
 assert o.param_groups[0]['lr']==expected
 v=s.state_dict();t=SceneExposureScheduler(o,SPEC,103288);t.load_state_dict(v)
 assert t.state_dict()==v
 t.set_exposures(103288);assert o.param_groups[0]['lr']<5e-5
 assert lr_multiplier(4.,SPEC)==.05
 assert lr_multiplier(.4,SPEC)==1.
