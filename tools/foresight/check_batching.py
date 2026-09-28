"""Actual Qwen3-VL images/DeepStack/mRoPE: padded batch vs separate FP32 inference."""
import argparse
import copy
from pathlib import Path
import subprocess
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','current-root','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Evidence is immutable')
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'real_Qwen_batch_padding','real_optimizer_updates':0}) as (meter,_,save):
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Use locked source')
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=DDPForesight(OmegaConf.load(a.config)).float().cuda().eval();model.inference_fp32=True
        data=ForesightCurrentDataset(a.current_root);examples=[data[i] for i in range(4)]
        # Current-text-only perturbation forces unequal prompt lengths. No label used.
        examples[-1]=dict(examples[-1],lang=examples[-1]['lang']+' Use the currently available observations.')
        tolerance={'atol':1e-3,'rtol':1e-4};report={'source_sha':source,'real_three_front_images':True,'precision':'FP32',
                     'tolerance':tolerance,'current_text_padding_probe':True,'comparisons':[]}
        with torch.inference_mode():
            together=model.encode_current(examples)
            for i,example in enumerate(examples):
                single=model.encode_current([example]);row={'sample_index':i}
                for key in ('W','action_queries'):
                    row[key+'_max_abs']=float((together[key][i:i+1]-single[key]).abs().max())
                    torch.testing.assert_close(together[key][i:i+1],single[key],**tolerance)
                report['comparisons'].append(row)
            noise=torch.randn(4,8,4,device='cuda',generator=torch.Generator(device='cuda').manual_seed(42))
            batched=model.predict_action(examples,initial_noise=noise)
            separate=torch.cat([model.predict_action([example],initial_noise=noise[i:i+1]) for i,example in enumerate(examples)])
            report['ego_max_abs']=float((batched-separate).abs().max());torch.testing.assert_close(batched,separate,**tolerance)
        report['passed']=True;report['peak_memory_bytes']=torch.cuda.max_memory_allocated();atomic_json(a.output,report)
        meter['inference_scenes']=4;save()


if __name__=='__main__':main()
