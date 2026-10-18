"""Full deployment bundle: trained Qwen/provider plus merged source action expert."""
import json
from pathlib import Path
import shutil
import torch
from .contracts import file_sha,QWEN_PATH,ACTION_CONFIG,runtime_identity

PROCESSOR_FILES=('chat_template.json','config.json','merges.txt','preprocessor_config.json',
    'tokenizer.json','tokenizer_config.json','video_preprocessor_config.json','vocab.json')


def package_processor(destination):
    """Copy only small pinned processor assets, never generic model weights."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in PROCESSOR_FILES:
        shutil.copyfile(QWEN_PATH/name,destination/name)
        hashes[name]=file_sha(destination/name)
    return hashes


def inspect_export(path):
    path=Path(path);seal=json.loads((path/'COMPLETE').read_text());info=json.loads((path/'export.json').read_text())
    if ('.incomplete' in path.name or seal['export_sha256']!=file_sha(path/'export.json') or
        file_sha(path/'model.pt')!=info['model_sha256'] or
        info['schema']!='native_Qwen_sourceflow_merged_deployment_v1'):
        raise ValueError('Unsealed native deployment export')
    return info


def load_standalone(path, *, device='cpu'):
    """No parent model, generic weights, teacher, training cache or scorer."""
    from starVLA.model.modules.qwen_sourceflow.model import QwenSourceFlowVLA
    from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
    from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
    path=Path(path);info=inspect_export(path)
    if info['runtime']!=runtime_identity() or info['action_config']!=ACTION_CONFIG:
        raise ValueError('Unqualified standalone model/runtime')
    if set(info['processor_files'])!=set(PROCESSOR_FILES):raise ValueError('Incomplete standalone processor')
    for name,sha in info['processor_files'].items():
        if file_sha(path/'processor'/name)!=sha:raise ValueError('Standalone processor changed: '+name)
    cfg=json.loads((path/'processor/config.json').read_text())
    model=QwenSourceFlowVLA(None,ActionDiT(**ACTION_CONFIG),checkpoint_config=cfg)
    state=torch.load(path/'model.pt',map_location='cpu',mmap=False,weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False),QwenObservationProcessor(path/'processor'),info


def load_export(path,model,parent):
    path=Path(path);info=inspect_export(path)
    if info['parent_model_sha256']!=parent['files']['model.pt']:
        raise ValueError('Unsealed/wrong-parent native RL deployment export')
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',mmap=False,weights_only=True),strict=True)
    return info
