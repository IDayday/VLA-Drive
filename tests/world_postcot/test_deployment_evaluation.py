"""CPU export/receipt contracts; these are not real Qwen/Navtest results."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor
from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
from starVLA.model.modules.qwen_world_postcot.post_action import OFFICIAL_METRIC_NAMES
from starVLA.rl.qwen_world_postcot.contracts import fingerprint, file_sha
from starVLA.rl.qwen_world_postcot.deployment import inspect_export, load_lora_adapter_state, EXPORT_SCHEMA
from starVLA.rl.qwen_world_postcot.rl_actor import CleanXRecipeActor, install_deployment_lora
from starVLA.rl.qwen_world_postcot.rl_runtime import expected_cleanx_recipe
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state
from tools.qwen_world_postcot.evaluate import validate_population, validate_feedback, sampling_role, validate_dev_binding
from tools.qwen_world_postcot.summarize import load, compare, summarize


def tiny_actor():
    return CleanXActor(hidden_dim=16, num_layers=2, num_heads=4, ffn_dim=32)


def test_real_native_adapter_names_strict_unmerged_roundtrip_and_new_process(tmp_path):
    torch.manual_seed(51)
    cfg = expected_cleanx_recipe()
    wrapped = CleanXRecipeActor(tiny_actor(), cfg)
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if name.endswith('lora_B') and parameter.requires_grad:
                parameter.normal_(std=.01)
    adapters = adapter_state(wrapped)
    assert adapters and all(name.startswith('action_expert.blocks.') for name in adapters)
    head = copy.deepcopy(wrapped.ref_action_expert)
    install_deployment_lora(head, cfg['grpo'])
    load_lora_adapter_state(head, adapters)
    assert set(head.state_dict()) == set(wrapped.action_expert.state_dict())
    assert all(torch.equal(head.state_dict()[name], value) for name, value in wrapped.action_expert.state_dict().items())
    memory, ego, noise = torch.randn(2, 7, 16), torch.randn(2, 8), torch.randn(2, 8, 3)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        context = head.prepare_context(memory, ego=ego)
        expected = CleanXFlowPolicy(head).sample_ode(context, noise)
    artifact = tmp_path/'tiny.pt'; output = tmp_path/'result.pt'
    torch.save(dict(state=head.state_dict(), grpo=cfg['grpo'], memory=memory, ego=ego, noise=noise), artifact)
    command = """import sys,torch
from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor
from starVLA.model.modules.qwen_world_postcot.flow import CleanXFlowPolicy
from starVLA.rl.qwen_world_postcot.rl_actor import install_deployment_lora
torch.set_num_threads(1)
p=torch.load(sys.argv[1],weights_only=True)
h=CleanXActor(hidden_dim=16,num_layers=2,num_heads=4,ffn_dim=32)
install_deployment_lora(h,p['grpo']);h.load_state_dict(p['state'],strict=True);h.eval()
with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
 c=h.prepare_context(p['memory'],ego=p['ego'])
 output=CleanXFlowPolicy(h).sample_ode(c,p['noise'])
torch.save(output,sys.argv[2])
"""
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    subprocess.run([sys.executable, '-c', command, str(artifact), str(output)], check=True, env=environment)
    torch.testing.assert_close(torch.load(output, weights_only=True), expected, rtol=0, atol=0)
    wrong = dict(adapters); name = next(iter(wrong)); wrong[name] = wrong[name][:1]
    with pytest.raises(ValueError, match='shape/dtype'):
        load_lora_adapter_state(head, wrong)
    wrong = dict(adapters); wrong[name] = wrong[name].bfloat16()
    with pytest.raises(ValueError, match='shape/dtype'):
        load_lora_adapter_state(head, wrong)
    with pytest.raises(ValueError, match='closure'):
        load_lora_adapter_state(head, {})


def unit_export(root, *, diagnostic_rl=False):
    from starVLA.rl.qwen_sourceflow.export import PROCESSOR_FILES
    root.mkdir(); (root/'processor').mkdir()
    for name in PROCESSOR_FILES:
        (root/'processor'/name).write_text('{}')
    torch.save({'unit_only': torch.zeros(1)}, root/'model.pt')
    meta = dict(schema=EXPORT_SCHEMA, model_sha256=file_sha(root/'model.pt'),
        prediction_type='x0', action_shape=[8, 3], LoRA_merge=False,
        no_teacher_future_candidate_metric_or_train_heads=True, predictive_memory_retained=True,
        processor={name:file_sha(root/'processor'/name) for name in PROCESSOR_FILES},
        IL_parent={'identity':{'qualification':False}},
        RL_parent={'purpose':'qualification_not_formal_initialization'} if diagnostic_rl else None,
        rl_lora={'unit_only':True} if diagnostic_rl else None, qualification=diagnostic_rl)
    seal_export(root, meta)
    return meta


def seal_export(root, meta):
    (root/'export.json').write_text(json.dumps(meta))
    (root/'COMPLETE').write_text(json.dumps({'export_sha256':file_sha(root/'export.json')}))


def test_export_receipt_preserves_rl_diagnostic_and_processor_identity(tmp_path):
    root=tmp_path/'export';meta=unit_export(root,diagnostic_rl=True)
    assert inspect_export(root)['qualification'] is True
    meta['qualification']=False;seal_export(root,meta)
    with pytest.raises(ValueError,match='qualification'):
        inspect_export(root)
    meta['qualification']=True;seal_export(root,meta)
    (root/'processor/config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='processor changed'):
        inspect_export(root)


def population():
    return dict(dev_tokens=['a','b'],navtest_tokens=['z'],token_logs={'a':'physical_a_0_100','b':'physical_b_0_100','z':'heldout_0_100'})


def write_evaluation(root, *, scores=(1.,1.), seed=2101):
    root.mkdir();(root/'scenes').mkdir()
    pop=population()
    identity=dict(schema='world_postcot_evaluation_v1',cohort='DEV',qualification=False,
        tokens=pop['dev_tokens'],count=2,token_sha256=fingerprint(pop['dev_tokens']),
        model_sha256='unit_model',data_sha256='data',seed=seed,protocol={'sha256':'canonical'},
        inference='single_candidate_cleanx_K10_shift5',numerics={'unit_only':True})
    identity['sha256']=fingerprint(identity);(root/'identity.json').write_text(json.dumps(identity))
    for token,score in zip(identity['tokens'],scores):
        trajectory=np.zeros((8,3),dtype=np.float32).tolist()
        metrics={name:1. for name in OFFICIAL_METRIC_NAMES};metrics['no_at_fault_collisions']=score;metrics['score']=score
        feedback=dict(status='OK',token=token,log=pop['token_logs'][token],protocol_sha256='canonical',
            requested_trajectory=trajectory,metrics=metrics,cache={'sha256':'context_'+token})
        row=dict(status='OK',token=token,log=pop['token_logs'][token],identity=identity['sha256'],
            requested_trajectory=trajectory,feedback=feedback)
        row['sha256']=fingerprint(row);(root/'scenes'/f'{token}.json').write_text(json.dumps(row))
    return identity


def test_fixed_population_invalid_labels_full_denominator_and_paired_safety(tmp_path):
    pop=population()
    assert validate_population(['a','b'],'DEV',pop)==['a','b']
    with pytest.raises(ValueError):validate_population(['a'],'DEV',pop)
    with pytest.raises(ValueError):validate_population(['a','b'],'DEV',pop,limit=1)
    with pytest.raises(ValueError):validate_population(['a','b'],'DEV',pop,qualification=True,limit=-1)
    reference=tmp_path/'base';trial=tmp_path/'trial'
    write_evaluation(reference);write_evaluation(trial,scores=(0.,1.))
    result=summarize(trial,reference,population=pop)
    assert result['scene_denominator']==2 and result['PDMS']==50
    assert result['paired']['high_to_zero']==1
    assert result['paired']['safety_transitions']['no_at_fault_collisions']['new_failure']==1
    assert result['paired']['physical_groups']==2
    row=json.loads((trial/'scenes/a.json').read_text());row['feedback']['token']='b';row.pop('sha256');row['sha256']=fingerprint(row)
    (trial/'scenes/a.json').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='join'):
        load(trial,population=pop)
    (trial/'scenes/a.json').unlink()
    with pytest.raises(ValueError,match='population'):
        load(trial,population=pop)


def test_feedback_exact_request_and_invalid_metrics_never_become_zero(tmp_path):
    root=tmp_path/'eval';write_evaluation(root)
    feedback=json.loads((root/'scenes/a.json').read_text())['feedback']
    options=dict(token='a',log=population()['token_logs']['a'],requested=np.zeros((8,3),np.float32),protocol_sha256='canonical')
    validate_feedback(feedback,**options)
    bad=copy.deepcopy(feedback);bad['requested_trajectory'][0][0]=1.
    with pytest.raises(ValueError,match='exact submitted'):
        validate_feedback(bad,**options)
    bad=copy.deepcopy(feedback);bad['metrics']['score']=float('nan')
    with pytest.raises(ValueError,match='Invalid official metric'):
        validate_feedback(bad,**options)
    baseline=tmp_path/'different_seed';write_evaluation(baseline,seed=0)
    left,rows=load(root,population=population());right,base=load(baseline,population=population())
    with pytest.raises(ValueError,match='seed'):
        compare(left,rows,right,base)


def test_only_declared_postselection_sampling_seeds_can_reuse_dev2101_binding(monkeypatch):
    import tools.qwen_world_postcot.summarize as summary_module
    identity=dict(cohort='DEV',qualification=False,model_sha256='model',protocol={'sha256':'canonical'},
        seed=2101,numerics={'fixed':True},inference='single_candidate_cleanx_K10_shift5',sha256='DEV_receipt')
    monkeypatch.setattr(summary_module,'load',lambda path:(identity,{}))
    assert sampling_role('NAVTEST',2102,True)=='post_selection_sampling_variation'
    result=validate_dev_binding('/unit/DEV',model_sha256='model',protocol_sha256='canonical',
        seed=2102,numerics={'fixed':True},role='post_selection_sampling_variation')
    assert result['DEV_selection_seed']==2101 and result['evaluation_sampling_seed']==2102
    for cohort,seed,variation in [('DEV',2102,True),('NAVTEST',999,True),('NAVTEST',2102,False),('NAVTEST',2101,True)]:
        with pytest.raises(ValueError):sampling_role(cohort,seed,variation)
    for seed,role,model in [(2102,'primary_seed2101','model'),(999,'post_selection_sampling_variation','model'),
                            (2103,'post_selection_sampling_variation','other')]:
        with pytest.raises(ValueError):validate_dev_binding('/unit/DEV',model_sha256=model,
            protocol_sha256='canonical',seed=seed,numerics={'fixed':True},role=role)


def test_explicit_legacy_adapter_keeps_canonical_scene_contract_while_allowing_named_method_changes(tmp_path):
    from tools.qwen_world_postcot.legacy_baseline import load_legacy,compare_registered,BINDING_SCHEMA
    from scripts.analysis.qwen_sourceflow.evaluation_integrity import verify_identity
    root=tmp_path/'legacy';(root/'scenes').mkdir(parents=True)
    pop=dict(navtest_tokens=['scene'],token_logs={'scene':'physical_0_19'})
    protocol=dict(name='navsim_v1.1_independent_reference_candidate');protocol['sha256']=fingerprint(protocol)
    identity=dict(schema='native_qwen_flow_evaluation_v1',metric='canonical_NAVSIM_v1.1',cohort='NAVTEST',
        seed=2101,inference='single_candidate_source_shift5_ODE_K10',protocol=protocol,qualification=False,
        count=1,token_sha256=fingerprint(pop['navtest_tokens']),data_sha256='old_train',precision='old documented precision',
        parent_model_sha256='old_model',checkpoint='/old/checkpoint',rl_export=None)
    identity['sha256']=fingerprint(identity);verify_identity(identity)
    (root/'identity.json').write_text(json.dumps(identity))
    metrics=dict(zip(OFFICIAL_METRIC_NAMES,[1.,1.,1.,.5,1.,1.]));metrics['score']=9.5/12
    feedback=dict(status='OK',token='scene',log=pop['token_logs']['scene'],protocol=protocol['name'],protocol_sha256=protocol['sha256'],
        requested_trajectory=np.zeros((8,3),np.float32).tolist(),metrics=metrics,cache={'sha256':'actual_context'})
    row=dict(status='OK',token='scene',log=pop['token_logs']['scene'],identity=identity['sha256'],
        requested_trajectory=feedback['requested_trajectory'],normalized_action=feedback['requested_trajectory'],feedback=feedback)
    row['sha256']=fingerprint(row);(root/'scenes/scene.json').write_text(json.dumps(row))
    summary=dict(status='COMPLETE',qualification=False,cohort='NAVTEST',full_population=True,seed=2101,
        evaluation=str(root),scenes=1,means=metrics,source_identity=identity['sha256'],model_sha256='old_model',
        IL_parent_checkpoint='/old/checkpoint',rl_export=None);summary['sha256']=fingerprint(summary)
    path=tmp_path/'native_summary.json';path.write_text(json.dumps(summary))
    entry=dict(name='native',updates=77000,model_sha256='old_model',summary=str(path),summary_file_sha256=file_sha(path),evaluation_identity_sha256=identity['sha256'],
        PDMS_reconstructed=100*metrics['score'],architecture={'prediction_type':'velocity'},precision=identity['precision'],
        data_exposure='documented',budget_and_parent_caveat='different parent')
    old,rows=load_legacy(entry,population=pop)
    binding=dict(schema=BINDING_SCHEMA,entries=[entry]);binding['sha256']=fingerprint(binding)
    pinned=tmp_path/'BASELINE_BINDING.json';pinned.write_text(json.dumps(binding))
    new=dict(schema='world_postcot_evaluation_v1',cohort='NAVTEST',qualification=False,seed=2101,
        inference='single_candidate_cleanx_K10_shift5',protocol=protocol,data_sha256='new_train')
    result=compare_registered(new,rows,pinned,population=pop)['native']
    assert result['mean_delta']==0. and result['physical_groups']==1
    assert result['explicitly_allowed_method_differences']['new_data_manifest_sha256']=='new_train'
    for update in ({'seed':2102},{'cohort':'DEV'},{'qualification':True}):
        with pytest.raises(ValueError,match='primary'):
            compare_registered(dict(new,**update),rows,pinned,population=pop)
    modified=copy.deepcopy(rows);modified['scene']['feedback']['cache']['sha256']='other_context'
    with pytest.raises(ValueError,match='reference context'):
        compare_registered(new,modified,pinned,population=pop)
