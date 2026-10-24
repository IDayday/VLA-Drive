"""One TRAIN-only gradient-scale calibration shared by S2 and S3."""
import argparse
from pathlib import Path
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import fingerprint
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol,require_budget_lock
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.feedback.student import load_teacher_bank, choose_teacher, teacher_normalized_target, independent_fm_draw


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True);p.add_argument('--scalar-bank',required=True)
    p.add_argument('--feedback-bank',required=True);p.add_argument('--output',required=True);p.add_argument('--scenes',type=int,default=16)
    a=p.parse_args();protocol=read_protocol(a.protocol);require_budget_lock(protocol);output=Path(a.output)
    if output.exists():raise FileExistsError('Calibrate once; preserve the original scale receipt')
    scalar,scalar_hash=load_teacher_bank(a.scalar_bank,protocol,expected_method=('F2_PDMS',))
    physical,physical_hash=load_teacher_bank(a.feedback_bank,protocol,expected_method=('F3_FEEDBACK','F4_NOVELTY'))
    env=NativeEnvironment(protocol,trainable='lora',scorer_workers=1,cache_scenes=a.scenes,
                          experiment_id='feedback_teacher_gradient_scale_TRAIN_only')
    tokens=protocol['splits']['TRAIN'][:a.scenes];params=[p for p in env.actor.parameters() if p.requires_grad]
    def gradients(kind):
        for p in params:p.grad=None
        eligible=0
        for exposure,token in enumerate(tokens):
            condition=env.condition(token,'TRAIN')
            if kind=='GT':
                noise,times=independent_fm_draw(42,exposure,'GT_scale_calibration')
                loss,_=env.source_fm_loss(condition,env.normalized_gt(token),noise=noise,times=times)
                (loss/len(tokens)).backward();env.ledger.add(model_backward_calls=1)
            else:
                for bank in [scalar,physical]:
                    row=choose_teacher(bank[token],42,exposure)
                    if row is None:continue
                    noise,times=independent_fm_draw(42,exposure,'teacher_scale_calibration')
                    loss,_=env.source_fm_loss(condition,teacher_normalized_target(row,env.codec),noise=noise,times=times)
                    (loss/(2*len(tokens))).backward();eligible+=1;env.ledger.add(model_backward_calls=1)
        values=[torch.zeros_like(p).cpu() if p.grad is None else p.grad.detach().cpu().clone() for p in params]
        return values,eligible
    try:
        gt,_=gradients('GT');teacher,eligible=gradients('teacher')
        norm_gt=float(torch.sqrt(sum(x.double().square().sum() for x in gt)))
        norm_teacher=float(torch.sqrt(sum(x.double().square().sum() for x in teacher)))
        if norm_gt<=0 or norm_teacher<=0:raise ValueError('No nonzero TRAIN teacher gradient; do not invent a calibrated beta')
        cosine=float(sum((x.double()*y.double()).sum() for x,y in zip(gt,teacher))/(norm_gt*norm_teacher))
        unbounded_beta=norm_gt/norm_teacher
        beta=min(2.,max(.05,unbounded_beta))
        record=dict(status='COMPLETE',schema='feedback_teacher_scale_v1',common_protocol_sha256=protocol['sha256'],
            base_model_sha256=protocol['base_model_sha256'],calibrated_split='TRAIN',scene_count=len(tokens),
            calibrated_scene_token_hash=fingerprint(tokens),scalar_teacher_bank_hash=scalar_hash,physical_teacher_bank_hash=physical_hash,
            eligible_method_scene_targets=eligible,GT_gradient_norm=norm_gt,teacher_gradient_norm=norm_teacher,
            GT_teacher_gradient_cosine=cosine,teacher_coefficient=beta,
            coefficient_rule='clip(GT_norm/combined_F2_F3_teacher_norm,0.05,2)',target_initial_gradient_ratio=1.,
            unbounded_teacher_coefficient=unbounded_beta,coefficient_bound_hit=beta!=unbounded_beta,
            fixed_scene_denominator=True,actor_optimizer_updates=0,ledger=env.ledger.snapshot())
        record['sha256']=fingerprint(record);atomic_json(output,record);print(record)
    finally:env.close()


if __name__=='__main__':main()
