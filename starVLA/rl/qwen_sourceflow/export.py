"""Full deployment bundle: trained Qwen/provider plus merged source action expert."""
import json
from pathlib import Path
import torch
from .contracts import file_sha


def load_export(path,model,parent):
    path=Path(path);seal=json.loads((path/'COMPLETE').read_text());info=json.loads((path/'export.json').read_text())
    if ('.incomplete' in path.name or seal['export_sha256']!=file_sha(path/'export.json') or
        info['parent_model_sha256']!=parent['files']['model.pt'] or
        file_sha(path/'model.pt')!=info['model_sha256'] or
        info['schema']!='native_Qwen_sourceflow_merged_deployment_v1'):
        raise ValueError('Unsealed/wrong-parent native RL deployment export')
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=True),strict=True)
    return info
