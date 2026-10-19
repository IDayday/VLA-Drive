"""Read-only check of actual action-query, visual and language weight updates."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
c=json.loads((ROOT/'assets.local.json').read_text());base=torch.load(c['checkpoint'],map_location='cpu',weights_only=True,mmap=True);mapping=json.loads((Path(c['tokenizer_root'])/'added_custom_token_id_map.json').read_text());ids=[mapping[f'<robot_action_{i}>'] for i in range(8)]
keys={'query':'qwen_vl_interface.model.model.language_model.embed_tokens.weight','visual':'qwen_vl_interface.model.model.visual.patch_embed.proj.weight','language':'qwen_vl_interface.model.model.language_model.layers.0.self_attn.q_proj.weight'};result={}
for name in sys.argv[1:]:
    root=ROOT/'runs'/name;receipt=json.loads((root/'RESULT.json').read_text());saved=torch.load(receipt['checkpoint'],map_location='cpu',mmap=True,weights_only=False)['model'];diff={}
    for label,key in keys.items():
        if label=='query':a=base[key][ids].float();b=saved[key][ids].float()
        else:a=base[key].float();b=saved[key].float()
        diff[label]=float((a-b).abs().max())
    needs_change=receipt['arm']!='C_DiT';assert all((v>0 if needs_change else v==0) for v in diff.values()),(name,diff)
    result[name]={'arm':receipt['arm'],'action_query_token_ids':ids,'max_abs_change':diff};del saved
(ROOT/'ACTION_QUERY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
