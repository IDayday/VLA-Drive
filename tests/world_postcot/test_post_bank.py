import json
from pathlib import Path

import numpy as np
import pytest
import torch

from starVLA.model.modules.qwen_world_postcot.post_action import (
    OFFICIAL_METRIC_NAMES, PostActionHead, choose_aux_scenes, post_action_loss,
    replay_post_action,
)
from starVLA.model.modules.qwen_world_postcot.candidate_bank import (
    CandidateBank, CandidateBankWriter, CandidateRecord, CandidatePairBatch, trajectory_hash,
    merge_bank_parts,
)
from tools.qwen_world_postcot.build_candidate_bank import coherent_proposals


def record(token='scene_a', variant=0, nc=1., dac=1.):
    poses = np.zeros((8, 3), dtype=np.float32)
    poses[:, 0] = np.arange(1, 9)*2+variant*.05
    labels = dict(zip(OFFICIAL_METRIC_NAMES, [nc, dac, 1., .75, 1., 1.]))
    return CandidateRecord(scene_id=token, source_log='physical_log', split='TRAIN',
        timestamp_us=1630000000000000, observation_hash='obs_'+token,
        trajectory_fp32=poses.tolist(), trajectory_sha256=trajectory_hash(poses),
        raw_factor_labels=labels, valid_masks={name: True for name in labels},
        raw_total_score=.9*nc*dac, score_protocol_sha256='protocol',
        evaluator_revision={'source_tree_sha256': 'official_source'}, metric_version='canonical_v1.1',
        context_sha256='cache_'+token, candidate_source='coherent_variant',
        actor_checkpoint_sha256=None, sampling={})


def make_bank(path, tokens=('scene_a', 'scene_b'), *, minimum=2):
    identity = dict(split='TRAIN', score_protocol_sha256='protocol', raw_manifest='raw')
    with CandidateBankWriter(path, identity, tokens, minimum_candidates=minimum,
                             shard_bytes=2048) as writer:
        for token in tokens:
            writer.append_many([record(token, 0), record(token, 1, nc=0.), record(token, 2, nc=.5)])
        writer.finalize()
    return identity


def test_factor_soft_labels_and_pair_direction():
    logits = torch.zeros((1, 2, 6), requires_grad=True)
    labels = torch.ones_like(logits)
    labels[:, 1, 0] = 0.
    labels[:, :, 1] = .5
    result = post_action_loss(logits, labels, torch.ones_like(labels, dtype=torch.bool), distributed=False)
    assert result['pair_count'] == 1
    result['pair'].backward()
    assert logits.grad[0, 0, 0] < 0 and logits.grad[0, 1, 0] > 0
    assert torch.equal(logits.grad[:, :, 1], torch.zeros_like(logits.grad[:, :, 1]))


def test_invalid_marker_is_not_an_ordinary_zero_and_nonfinite_prediction_fails():
    logits = torch.zeros((1, 2, 6), requires_grad=True)
    labels = torch.full_like(logits, float('nan'))
    valid = torch.zeros_like(labels, dtype=torch.bool)
    result = post_action_loss(logits, labels, valid, distributed=False)
    assert result['loss'] == 0 and result['factor_count'] == 0
    result['loss'].backward()
    assert torch.isfinite(logits.grad).all()
    logits = logits.detach(); logits[0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        post_action_loss(logits, labels, valid, distributed=False)


def test_full_update_normalizers_make_microbatch_gradients_identical():
    generator = torch.Generator().manual_seed(32)
    initial = torch.randn((3, 2, 6), generator=generator)
    labels = torch.ones_like(initial)
    labels[0, 1, 0] = 0.; labels[2, 0, 1] = 0.
    valid = torch.ones_like(initial, dtype=torch.bool)
    valid[1] = False
    whole = initial.clone().requires_grad_()
    loss = post_action_loss(whole, labels, valid, distributed=False)['loss']
    loss.backward()
    micros = initial.clone().requires_grad_()
    for i in range(3):
        partial = post_action_loss(micros[i:i+1], labels[i:i+1], valid[i:i+1],
            distributed=False, scene_denominators={'factor': 2., 'pair': 2.})
        partial['loss'].backward()
    # Separate CPU BCE launches can differ by one FP32 ULP. This verifies the
    # objective/denominators, not a bitwise promise across launch shapes.
    torch.testing.assert_close(micros.grad, whole.grad, rtol=1e-6, atol=1e-8)


def test_head_candidate_permutation_and_duplicate_invariance():
    torch.manual_seed(7)
    head = PostActionHead(input_dim=16, width=16, num_heads=4).eval()
    hidden = torch.randn(3, 8, 16)
    expected = head(hidden)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(head(hidden[permutation]), expected[permutation])
    duplicate = head(hidden[[1, 1]])
    torch.testing.assert_close(duplicate[0], duplicate[1], rtol=0, atol=0)


def test_aux_selection_uniform_and_restorable():
    generator = torch.Generator().manual_seed(42)
    state = generator.get_state()
    first = choose_aux_scenes(['a', 'b', 'c', 'd'], generator=generator)
    generator.set_state(state)
    assert torch.equal(first, choose_aux_scenes(['a', 'b', 'c', 'd'], generator=generator))
    counts = torch.zeros(4)
    for _ in range(1024):
        counts[choose_aux_scenes(['a', 'b', 'c', 'd'], generator=generator)] += 1
    assert ((counts > 420) & (counts < 610)).all()


def test_bank_exact_coordinates_softlabels_and_sampler_resume(tmp_path):
    identity = make_bank(tmp_path/'bank')
    bank = CandidateBank(tmp_path/'bank', expected_identity=identity, seed=13)
    assert bank.manifest['record_count'] == 6
    assert any(row.raw_factor_labels['no_at_fault_collisions'] == .5 for row in bank.records('scene_a'))
    state = bank.state_dict()
    one = bank.sample_pairs(['scene_a', 'scene_b'])
    bank.load_state_dict(state)
    two = bank.sample_pairs(['scene_a', 'scene_b'])
    assert one.record_ids == two.record_ids and one.pair_mask.all(dim=0).tolist() == [True, False]
    torch.testing.assert_close(one.trajectory, two.trajectory, rtol=0, atol=0)
    assert not one.trajectory.requires_grad and not one.labels.requires_grad
    with pytest.raises(ValueError, match='observation hash'):
        bank.records('scene_a', observation_hash='wrong')
    bank.close()


def test_missing_partial_and_role_mismatch_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        CandidateBank(tmp_path/'missing')
    identity = make_bank(tmp_path/'bank')
    with pytest.raises(ValueError, match='role'):
        CandidateBank(tmp_path/'bank', allowed_splits=('DEV',))
    with pytest.raises(ValueError, match='identity mismatch'):
        CandidateBank(tmp_path/'bank', expected_identity=dict(raw_manifest='changed'))
    with CandidateBankWriter(tmp_path/'partial', identity, ['scene_a']) as writer:
        writer.append_many([record()])
        with pytest.raises(ValueError, match='Incomplete'):
            writer.finalize()
    assert not (tmp_path/'partial/COMPLETE.json').exists()


def test_uncommitted_shard_tail_is_preserved_and_resume_idempotent(tmp_path):
    root = tmp_path/'bank'; identity = dict(split='TRAIN', score_protocol_sha256='protocol')
    with CandidateBankWriter(root, identity, ['scene_a']) as writer:
        writer.append_many([record()])
    shard = next((root/'shards').glob('*.jsonl'))
    with shard.open('ab') as handle:
        handle.write(b'incomplete_record_tail')
    with CandidateBankWriter(root, identity, ['scene_a'], resume=True) as writer:
        assert writer.append_many([record()]) == dict(added=0, reused=1)
        writer.append_many([record(variant=1)])
        writer.finalize()
    assert list(root.glob('UNCOMMITTED_*'))
    bank = CandidateBank(root)
    assert len(bank.records('scene_a')) == 2
    bank.close()


def test_corruption_is_rejected_even_after_scene_was_cached(tmp_path):
    make_bank(tmp_path/'bank')
    bank = CandidateBank(tmp_path/'bank')
    bank.records('scene_a')
    shard = next((tmp_path/'bank/shards').glob('*.jsonl'))
    with shard.open('r+b') as stream:
        stream.seek(10); stream.write(b'X')
    with pytest.raises(ValueError, match='changed|checksum|modified'):
        bank.records('scene_a')
    bank.close()


def test_coherent_variants_real_shape_stationary_and_zero_identity():
    for gt in [np.zeros((8, 3), dtype=np.float32),
               np.stack([np.arange(1, 9)*2, np.zeros(8), np.zeros(8)], axis=1).astype(np.float32)]:
        values, _ = coherent_proposals(gt, 16)
        assert len(values) == 16
        np.testing.assert_array_equal(np.asarray(values[0]['trajectory_fp32']), gt)
        assert len({trajectory_hash(value['trajectory_fp32']) for value in values}) == 16
        assert all(value['dynamics_check']['valid'] for value in values)


def test_nonfp32_coordinates_require_rescoring():
    trajectory = np.ones((8, 3), dtype=np.float64)*.1
    with pytest.raises(ValueError, match='not exact FP32'):
        trajectory_hash(trajectory)


def test_real_cleanx_actor_post_only_gradient_contract():
    from starVLA.model.modules.qwen_world_postcot.actor import CleanXActor
    torch.manual_seed(9)
    actor = CleanXActor(hidden_dim=16, num_layers=2, num_heads=4, ffn_dim=32)
    head = PostActionHead(input_dim=16, width=16, num_heads=4)
    optimizer = torch.optim.AdamW([*actor.parameters(), *head.parameters()], lr=.002)
    memory = torch.randn(2, 5, 16, requires_grad=True)
    ego = torch.randn(2, 8)
    coordinates = torch.tensor([[record('a', 0).trajectory_fp32, record('a', 1).trajectory_fp32],
                                [record('b', 0).trajectory_fp32, record('b', 1).trajectory_fp32]],
                               requires_grad=True)
    labels = torch.ones(2, 2, 6, requires_grad=True)
    with torch.no_grad():
        labels[:, 1, 0] = 0.
    pair = CandidatePairBatch(coordinates, labels, torch.ones_like(labels, dtype=torch.bool),
        torch.tensor([[True, False], [True, False]]), [['a0', 'a1'], ['b0', 'b1']], ['a', 'b'], 'test')
    output_calls = []
    hook = actor.action_output.register_forward_hook(lambda *args: output_calls.append(True))
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True); memory.grad = None
        context = actor.prepare_context(memory, ego=ego)
        loss = replay_post_action(actor, head, context, torch.tensor([0, 1]), pair, distributed=False)['loss']
        loss.backward()
        assert memory.grad is not None and memory.grad.abs().sum() > 0
        assert actor.action_input.weight.grad.abs().sum() > 0
        assert actor.blocks[0].cross_attn.q.weight.grad.abs().sum() > 0
        assert actor.action_output.weight.grad is None
        assert coordinates.grad is None and labels.grad is None
        optimizer.step()
    # AdaLN-zero is expected to gate FFN/self gradients on the first update;
    # post-only training unlocks them rather than leaving an isolated scorer.
    assert actor.blocks[0].ffn[0].weight.grad.abs().sum() > 0
    assert actor.blocks[0].self_attn.q.weight.grad.abs().sum() > 0
    assert not output_calls
    hook.remove()


def _gloo_post_worker(rank, init_file, output, state, features, labels, valid):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+init_file, rank=rank, world_size=2)
    try:
        module = torch.nn.Linear(6, 6, bias=False)
        module.load_state_dict(state, strict=True)
        model = DistributedDataParallel(module)
        ids = slice(0, 2) if rank == 0 else slice(2, 3)
        logits = model(features[ids])
        post_action_loss(logits, labels[ids], valid[ids], distributed=True)['loss'].backward()
        torch.save(module.weight.grad, Path(output)/f'gloo_rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_actual_cpu_gloo_ddp_matches_global_loss_with_empty_label_rank(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('CPU/Gloo unavailable')
    torch.manual_seed(41)
    module = torch.nn.Linear(6, 6, bias=False)
    features = torch.randn(3, 2, 6)
    labels = torch.ones(3, 2, 6); labels[0, 1, 0] = 0.
    valid = torch.ones_like(labels, dtype=torch.bool); valid[2] = False
    post_action_loss(module(features), labels, valid, distributed=False)['loss'].backward()
    expected = module.weight.grad.clone()
    mp.spawn(_gloo_post_worker, args=(str(tmp_path/'gloo_init'), str(tmp_path),
        module.state_dict(), features, labels, valid), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path/f'gloo_rank{rank}.pt', weights_only=True)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def test_manifest_commit_window_and_disjoint_parts_merge(tmp_path):
    root = tmp_path/'bank'; identity = dict(split='TRAIN', score_protocol_sha256='protocol')
    parts = []
    for i, token in enumerate(['scene_a', 'scene_b']):
        part = root/'parts'/f'rank{i:05d}'; parts.append(part)
        with CandidateBankWriter(part, identity, [token]) as writer:
            writer.append_many([record(token, 0), record(token, 1)])
            writer.finalize()
    manifest = merge_bank_parts(root, parts, identity=identity, scene_ids=['scene_a', 'scene_b'])
    assert manifest['scene_count'] == 2 and manifest['record_count'] == 4
    assert merge_bank_parts(root, parts, identity=identity, scene_ids=['scene_a', 'scene_b']) == manifest
    bank = CandidateBank(root)
    assert len(bank.records('scene_b')) == 2
    bank.close()


def test_retry_after_manifest_before_complete_does_not_overwrite(tmp_path):
    root = tmp_path/'bank'; identity = dict(split='TRAIN', score_protocol_sha256='protocol')
    with CandidateBankWriter(root, identity, ['scene_a']) as writer:
        writer.append_many([record('scene_a', 0), record('scene_a', 1)])
        original = writer.finalize()
    (root/'COMPLETE.json').unlink()  # explicit unit-test fault injection only
    with CandidateBankWriter(root, identity, ['scene_a'], resume=True) as writer:
        recovered = writer.finalize()
    assert original == recovered


def test_reusable_labels_require_exact_current_context_and_observation():
    from tools.qwen_world_postcot.build_candidate_bank import reusable_feedback
    value = record('scene_a', 0)
    context = dict(path='/unit-only-context', sha256=value.context_sha256)
    kwargs = dict(observation_hash=value.observation_hash, token=value.scene_id,
                  log=value.source_log, protocol={'sha256': value.score_protocol_sha256}, context=context)
    reused = reusable_feedback(value, **kwargs)
    assert reused['requested_trajectory'] == value.trajectory_fp32
    assert reused['metrics']['score'] == value.raw_total_score
    assert reusable_feedback(value, **dict(kwargs, observation_hash='different')) is None
    assert reusable_feedback(value, **dict(kwargs, context=dict(context, sha256='changed'))) is None


def test_original_gt_is_retained_for_real_scoring_when_motion_prefilter_flags_it():
    from tools.qwen_world_postcot.build_candidate_bank import coherent_proposals
    gt = np.zeros((8, 3), dtype=np.float32)
    gt[:, 0] = np.arange(1, 9, dtype=np.float32) * 10
    gt[2, 0] += 5  # a recorded target requiring audit, not a safe intervention
    proposals, _ = coherent_proposals(gt, 16)
    assert proposals[0]['source'] == 'logged_gt'
    np.testing.assert_array_equal(proposals[0]['trajectory_fp32'], gt)
    assert not proposals[0]['dynamics_check']['valid']


def test_logged_pose_timing_gap_is_candidate_provenance_not_pretended_gt():
    from tools.qwen_world_postcot.build_candidate_bank import logged_candidate_provenance
    row=dict(raw_future_dt_seconds=[.5,.5,1.,.5,.5,.5,.5,.5],timestamp_gap=True)
    value=logged_candidate_provenance(row)
    assert value['source_semantics']=='logged_pose_candidate_time_mismatch'
    assert value['raw_future_dt_seconds']==row['raw_future_dt_seconds']
    assert value['actual_official_request_times_seconds']==[.5,1.,1.5,2.,2.5,3.,3.5,4.]
    assert value['candidate_scoring_does_not_certify_imitation_target_timing']


def test_full_record_audit_checks_canonical_labels_and_timing_overlay(tmp_path):
    from dataclasses import replace
    from tools.qwen_world_postcot.audit_candidate_labels import _audit_production_shard
    root=tmp_path/'bank';identity=dict(split='TRAIN',score_protocol_sha256='protocol')
    original=replace(record('scene_a',0),candidate_source='logged_gt',raw_total_score=(5*.75+5+2)/12)
    alternate=replace(record('scene_a',1,nc=0.),raw_total_score=0.)
    with CandidateBankWriter(root,identity,['scene_a'])as writer:
        writer.append_many([original,alternate]);manifest=writer.finalize()
    name,info=next(iter(manifest['shards'].items()))
    timing={'scene_a':dict(scene_id='scene_a',trajectory_sha256=original.trajectory_sha256,
        timestamp_gap=True,raw_future_dt_seconds=[.5,.5,1.,.5,.5,.5,.5,.5])}
    import sqlite3
    with sqlite3.connect(root/'index.sqlite3')as index:
        indexed=list(index.execute('SELECT offset,length,checksum,record_id FROM records WHERE shard=? ORDER BY offset',(name,)))
    result=_audit_production_shard((str(root),name,info,timing,indexed))
    assert result['sources']=={'logged_gt':1,'coherent_variant':1}
    assert result['boundaries']['scene_a']&3==3
    assert result['corrections'][0]['source_semantics']=='logged_pose_candidate_time_mismatch'
    assert result['corrections'][0]['valid_official_candidate'] is True
    assert result['corrections'][0]['qualified_fixed_dt_imitation_target'] is False
    assert result['every_record_index_offset_checksum_verified']


def test_merge_native_indices_deduplicates_labels_without_replaying_simulation(tmp_path):
    import sqlite3
    from types import SimpleNamespace
    from tools.qwen_world_postcot.audit_candidate_labels import merge_native_indices
    from starVLA.model.modules.qwen_world_postcot.candidate_bank import atomic_json, file_sha
    parts = []
    for rank in range(2):
        path = tmp_path/f'part{rank}.sqlite'; parts.append(path)
        db = sqlite3.connect(path)
        db.execute('CREATE TABLE candidates(record_id TEXT PRIMARY KEY,token TEXT,payload TEXT)')
        db.execute('CREATE TABLE completed_files(path TEXT PRIMARY KEY,sha256 TEXT,eligible INTEGER)')
        db.execute('INSERT INTO candidates VALUES(?,?,?)', ('identical-request','scene_a',json.dumps({'feedback':{'score':0.5}})))
        db.execute('INSERT INTO completed_files VALUES(?,?,?)', (f'/unit-only-source{rank}','sha',1))
        db.commit(); db.close()
        info = dict(schema='unit-test-native-index',split='TRAIN',world=2,rank=rank,
            source_data_manifest_sha256='train',score_protocol_sha256='protocol',all_files_requested=True,
            source_file_budget=None,runs=[dict(run='/unit-run',identity_sha256='run',source_SFT_sha256='sft')])
        atomic_json(path.with_suffix('.COMPLETE.json'),dict(status='COMPLETE',identity=info,index_sha256=file_sha(path)))
    destination=tmp_path/'merged.sqlite'
    assert merge_native_indices(SimpleNamespace(output=str(destination),part=[str(path) for path in parts]))==0
    with sqlite3.connect(destination) as db:
        assert db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM completed_files').fetchone()[0]==2


def test_refresh_retains_gt_own_history_without_score_selection_and_rejects_diagnostic_registry(tmp_path):
    from dataclasses import replace
    from tools.qwen_world_postcot.refresh_candidates import retain_history,publish_registry,RECIPE
    records=[replace(record(variant=i),raw_total_score=i/20)for i in range(20)]
    records[19]=replace(records[19],candidate_source='logged_gt',raw_total_score=0.)
    records[18]=replace(records[18],candidate_source='native_qwen_rl')
    selected=retain_history(records)
    assert len(selected)==14 and selected[0]is records[19] and selected[1]is records[18]
    assert len({r.trajectory_sha256 for r in retain_history(records+[records[0]])})==14
    identity=dict(split='TRAIN',score_protocol_sha256='protocol',refresh_recipe=RECIPE,
        qualification=True,refresh_schema='world_postcot_current_student_refresh_DIAGNOSTIC_v1')
    root=tmp_path/'diagnostic'
    with CandidateBankWriter(root,identity,['scene_a'],minimum_candidates=2)as writer:
        writer.append_many([record(variant=0),record(variant=1)]);writer.finalize()
    with pytest.raises(ValueError,match='complete registered'):
        publish_registry(root,tmp_path/'REGISTRY.json')
    assert not(tmp_path/'REGISTRY.json').exists()


def test_refresh_pins_consumed_checkpoint_bank_and_does_not_accept_formal_qualification(tmp_path):
    from copy import deepcopy
    from starVLA.model.modules.qwen_world_postcot.candidate_bank import fingerprint,file_sha
    from tools.qwen_world_postcot.refresh_candidates import snapshot_binding
    identity=dict(schema='world_postcot_full_IL_v1',uses_rl=False,qualification=False)
    root=tmp_path/'update010000';root.mkdir()
    state=root/'rank00_state.pt';torch.save(dict(builder=dict(schema='world_postcot_batch_builder_v2',
        bank_root='/fixed/consumed/bank',bank_hash='actual_saved_bank_hash')),state)
    (root/'model').write_bytes(b'sealed own model')
    files={'rank00_state.pt':file_sha(state),'model':file_sha(root/'model')}
    (root/'identity.json').write_text(json.dumps(identity));(root/'files.json').write_text(json.dumps(files))
    seal=dict(status='COMPLETE',updates=10000,files_sha256=file_sha(root/'files.json'),identity_sha256=fingerprint(identity),model_file='model')
    (root/'COMPLETE').write_text(json.dumps(seal))
    metadata=dict(qualification=False,RL_parent=None,IL_parent=dict(path=str(root),model_sha256=files['model'],identity=identity),model_sha256='export_model')
    binding=snapshot_binding(root,metadata)
    assert binding['parent_bank_root']=='/fixed/consumed/bank' and binding['parent_bank_hash']=='actual_saved_bank_hash'
    bad=deepcopy(metadata);bad['IL_parent']['identity']['qualification']=True
    with pytest.raises(ValueError,match='exact pinned'):
        snapshot_binding(root,bad)
    seal['updates']=32;identity['qualification']=True
    (root/'identity.json').write_text(json.dumps(identity));seal['identity_sha256']=fingerprint(identity)
    (root/'COMPLETE').write_text(json.dumps(seal));metadata['qualification']=True
    assert snapshot_binding(root,metadata,qualification=True)['qualification']
    with pytest.raises(ValueError,match='nine real formal'):
        snapshot_binding(root,metadata)
