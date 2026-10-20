"""Actual production class at reduced CPU width; no learned S0 payload read."""
import builtins,copy,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
from omegaconf import OmegaConf
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model


def fixture():
    return OmegaConf.create({'framework':{'qwenvl':{'vl_hidden_dim':12},'action_model':{
        'hidden_size':16,'DiTConfig':{'input_embedding_dim':16,'num_layers':2,'num_attention_heads':2,'attention_head_dim':8},
        'diffusion_model_cfg':{'output_dim':16,'cross_attention_dim':16,'dropout':.2,'final_dropout':True,'interleave_self_attention':True},
        'action_dim':4,'action_horizon':8,'num_inference_timesteps':10,'noise_beta_alpha':1.5,'noise_beta_beta':1.,'noise_s':.999,
        'num_timestep_buckets':1000,'repeated_diffusion_steps':8,'add_pos_embed':True,'max_seq_len':8}}})


def worker(artifact,result):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if ('model.modules.vlm.QWen3' in name or 'modeling_qwen3_vl' in name or name.startswith('navsim')):
            raise AssertionError('Unnecessary Qwen/NAVSIM load in head-only reconstruction')
        return original(name,*args,**kwargs)
    builtins.__import__=guarded
    p=torch.load(artifact,weights_only=True,map_location='cpu')
    model=get_action_model(OmegaConf.create(p['config'])).float().eval()
    model.load_state_dict(p['state_dict'],strict=True)
    with torch.no_grad():out=model.predict_action(p['condition'],initial_noise=p['noise'])
    torch.save(out,result)


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    model=get_action_model(fixture()).float().eval()
    cond=torch.randn(2,8,12);noise=torch.randn(2,8,4)
    with torch.no_grad():expected=model.predict_action(cond,initial_noise=noise)
    with tempfile.TemporaryDirectory(prefix='effect_head_cpu_') as tmp:
        tmp=Path(tmp);art=tmp/'head.pt';out=tmp/'output.pt'
        torch.save({'config':OmegaConf.to_container(model.full_config,resolve=True),
                    'state_dict':model.state_dict(),'condition':cond,'noise':noise},art)
        result=subprocess.run([sys.executable,__file__,'worker',str(art),str(out)],
            timeout=60,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'',
              'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'})
        assert result.returncode==0,result.stderr
        assert torch.equal(expected,torch.load(out,weights_only=True))
        return {'schema':'action_effect_head_cpu_roundtrip_review_v1','status':'PASS_REDUCED_WIDTH_PRODUCTION_HEAD',
                'strict_state_load':True,'new_process_euler10_exact':True,'Qwen_loaded':False,'NAVSIM_loaded':False,
                'actual_pretrained_S0_artifact_loaded':False,'new_GPU_calls':0,'simulator_calls':0,
                'action_head_source_sha256':hashlib.sha256(Path('starVLA/model/modules/action_model/GR00T_ActionHeader.py').read_bytes()).hexdigest()}


if __name__=='__main__':
    if len(sys.argv)>1:worker(sys.argv[2],sys.argv[3])
    else:print(json.dumps(main(),sort_keys=True,indent=2))
