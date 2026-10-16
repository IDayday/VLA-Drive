"""CPU fixture tests for admission and byte bounds; no real data/model jobs."""
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.st_world_policy import cache_storage_budget as budget


MIB = 1 << 20
LEDGER = 64 << 10


def write_json(path, value):
    path.write_text(json.dumps(value))


def claim(root, index=0, *, output=MIB, staging=MIB, metadata=MIB):
    return {"claim_id": "claim-" + str(index), "root": str(root),
            "cache_identity": hashlib.sha256(str(index).encode()).hexdigest(),
            "output_bytes": output, "staging_bytes": staging, "metadata_bytes": metadata}


def policy_file(tmp_path, claims=None, *, cap=32*MIB, reserve=0, legacy=None):
    path = tmp_path / "policy.json"
    value = {"schema": budget.SCHEMA, "campaign": "test-campaign",
             "state_dir": str(tmp_path / "ledger"), "allowed_roots": [str(tmp_path / "cache")],
             "cache_cap_bytes": cap+LEDGER, "free_reserve_bytes": reserve, "ledger_reserve_bytes": LEDGER,
             "legacy_reservations": legacy or [],
             "claims": claims if claims is not None else [claim(tmp_path / "cache" / "a")]}
    write_json(path, value)
    return path


def reserve(path, row=None):
    row = row or json.loads(path.read_text())["claims"][0]
    return budget.StorageBudget(path).reserve(row["root"], row["cache_identity"],
        **{k: row[k] for k in ("output_bytes", "staging_bytes", "metadata_bytes")})


def state(path):
    return json.loads((Path(json.loads(path.read_text())["state_dir"]) / "CLAIMS.json").read_text())


def _child_reserve(args):
    path, row = args
    try:
        return "OK", reserve(Path(path), row).receipt
    except budget.StorageBudgetError as exc:
        return "BLOCKED", str(exc)


def test_default_cap_and_separate_reserve(tmp_path):
    path = policy_file(tmp_path)
    value = json.loads(path.read_text())
    del value["cache_cap_bytes"], value["free_reserve_bytes"]
    write_json(path, value)
    policy = budget.StorageBudget(path)
    assert policy.core["cache_cap_bytes"] == 768 * budget.GIB
    assert policy.core["free_reserve_bytes"] == 1024 * budget.GIB
    assert policy.core["cache_cap_bytes"] < policy.core["free_reserve_bytes"]


def test_claim_is_idempotent_and_reserved_once(tmp_path):
    path = policy_file(tmp_path, cap=3*MIB)
    first = reserve(path)
    for _ in range(3):
        assert reserve(path).receipt == first.receipt
    assert len(state(path)["claims"]) == 1
    assert first.receipt["campaign_reserved_bytes"] == 3*MIB+LEDGER
    assert not first.root.exists()  # Admission itself does not start the writer.


def test_concurrent_admission_cannot_oversubscribe(tmp_path):
    rows = [claim(tmp_path / "cache" / str(i), i) for i in range(6)]
    path = policy_file(tmp_path, rows, cap=6*MIB)
    with multiprocessing.get_context("spawn").Pool(6) as pool:
        results = pool.map(_child_reserve, [(str(path), row) for row in rows])
    assert [r[0] for r in results].count("OK") == 2
    assert len(state(path)["claims"]) == 2
    assert all("aggregate" in r[1] for r in results if r[0] == "BLOCKED")


def test_concurrent_same_root_reuses_claim(tmp_path):
    path = policy_file(tmp_path, cap=3*MIB)
    row = json.loads(path.read_text())["claims"][0]
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        results = pool.map(_child_reserve, [(str(path), row)] * 8)
    assert all(r[0] == "OK" for r in results)
    assert len(state(path)["claims"]) == 1
    assert {r[1]["campaign_reserved_bytes"] for r in results} == {3*MIB+LEDGER}


def test_declared_capacity_and_aggregate_cap_are_both_checked(tmp_path):
    path = policy_file(tmp_path)
    row = json.loads(path.read_text())["claims"][0]
    with pytest.raises(budget.StorageBudgetError, match="allowance is too small"):
        budget.StorageBudget(path).reserve(row["root"], row["cache_identity"],
            output_bytes=MIB+1, staging_bytes=MIB, metadata_bytes=MIB)
    value = json.loads(path.read_text()); value["cache_cap_bytes"] = 3*MIB-1
    write_json(path, value)
    with pytest.raises(budget.StorageBudgetError, match="aggregate"):
        reserve(path)
    assert not (tmp_path / "cache").exists()


def test_changed_admitted_claim_envelope_or_identity_is_refused(tmp_path):
    path = policy_file(tmp_path); reserve(path)
    original = json.loads(path.read_text())
    value = copy.deepcopy(original); value["claims"][0]["metadata_bytes"] += 1
    write_json(path, value)
    with pytest.raises(budget.StorageBudgetError, match="claim changed"):
        reserve(path)
    value = copy.deepcopy(original); value["cache_cap_bytes"] += 1
    write_json(path, value)
    with pytest.raises(budget.StorageBudgetError, match="different policy"):
        reserve(path)
    write_json(path, original)
    with pytest.raises(budget.StorageBudgetError, match="unknown cache root or identity"):
        budget.StorageBudget(path).reserve(original["claims"][0]["root"], "f"*64,
            output_bytes=0, staging_bytes=0, metadata_bytes=0)


def test_additional_declarations_preserve_existing_claims(tmp_path):
    path = policy_file(tmp_path); reserve(path)
    value = json.loads(path.read_text()); second = claim(tmp_path / "cache" / "b", 1)
    value["claims"].append(second); write_json(path, value)
    reserve(path, second)
    assert len(state(path)["claims"]) == 2


@pytest.mark.parametrize("change", ["duplicate_root", "duplicate_identity", "nested", "escape", "dotdot"])
def test_ambiguous_and_escaping_claim_paths_are_refused(tmp_path, change):
    a = claim(tmp_path / "cache" / "a")
    b = claim(tmp_path / "cache" / "b", 1)
    if change == "duplicate_root": b["root"] = a["root"]
    if change == "duplicate_identity": b["cache_identity"] = a["cache_identity"]
    if change == "nested": b["root"] = a["root"] + "/child"
    if change == "escape": b["root"] = str(tmp_path / "outside")
    if change == "dotdot": b["root"] = str(tmp_path / "cache" / "a" / ".." / "b")
    path = policy_file(tmp_path, [a,b])
    with pytest.raises(budget.StorageBudgetError): budget.StorageBudget(path)


def test_symlink_unknown_path_and_unknown_existing_identity_are_refused(tmp_path):
    path = policy_file(tmp_path)
    with pytest.raises(budget.StorageBudgetError, match="unknown cache root"):
        budget.StorageBudget(path).reserve(tmp_path / "cache" / "unknown", "a"*64,
            output_bytes=0, staging_bytes=0, metadata_bytes=0)
    root = tmp_path / "cache" / "a"; root.parent.mkdir(); root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(budget.StorageBudgetError, match="symlink"):
        reserve(path)
    root.unlink(); root.mkdir(); (root / "foreign_file").write_text("preserve this")
    with pytest.raises(budget.StorageBudgetError, match="no cache identity"):
        reserve(path)
    assert (root / "foreign_file").read_text() == "preserve this"


def test_existing_cache_identity_is_verified_before_admission(tmp_path):
    identity = {"schema": "fixture"}; identity["identity"] = budget._digest(identity)
    row = claim(tmp_path / "cache" / "a"); row["cache_identity"] = identity["identity"]
    path = policy_file(tmp_path, [row]); root = Path(row["root"]); root.mkdir(parents=True)
    write_json(root / "identity.json", identity); reserve(path)
    identity["schema"] = "changed"; write_json(root / "identity.json", identity)
    with pytest.raises(budget.StorageBudgetError, match="existing cache identity"):
        reserve(path)


def test_interrupted_files_count_and_retry_reconciles_without_deletion(tmp_path):
    path = policy_file(tmp_path); handle = reserve(path); handle.root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="injected"):
        with handle.writing(MIB):
            (handle.root / "shard_000000.123.tmp").write_bytes(b"x"*8192)
            raise RuntimeError("injected process failure")
    assert state(path)["claims"][str(handle.root)]["charged_bytes"] == MIB
    resumed = reserve(path)
    info = (handle.root / "shard_000000.123.tmp").stat()
    assert state(path)["claims"][str(handle.root)]["charged_bytes"] == max(info.st_size,info.st_blocks*512)
    with resumed.writing(MIB):
        (handle.root / "shard_000000.orphan_1.safetensors").write_bytes(b"y"*8192)
    assert (handle.root / "shard_000000.123.tmp").exists()
    assert (handle.root / "shard_000000.orphan_1.safetensors").exists()


def test_per_write_staging_and_cap_stop_before_io(tmp_path):
    path = policy_file(tmp_path); handle = reserve(path)
    with pytest.raises(budget.StorageBudgetError, match="bounded staging"):
        with handle.writing(MIB+1): pytest.fail("write must not run")
    for _ in range(3):
        with handle.writing(MIB): pass
    with pytest.raises(budget.StorageBudgetError, match="reserved output"):
        with handle.writing(1): pytest.fail("write must not run")
    assert state(path)["claims"][str(handle.root)]["charged_bytes"] == 3*MIB


def test_disk_floor_admission_and_every_write(tmp_path, monkeypatch):
    path = policy_file(tmp_path, reserve=10*MIB)
    monkeypatch.setattr(budget.shutil, "disk_usage", lambda _: SimpleNamespace(free=13*MIB+LEDGER-1))
    with pytest.raises(budget.StorageBudgetError, match="free-space reserve"):
        reserve(path)
    monkeypatch.setattr(budget.shutil, "disk_usage", lambda _: SimpleNamespace(free=13*MIB+LEDGER))
    handle = reserve(path)
    monkeypatch.setattr(budget.shutil, "disk_usage", lambda _: SimpleNamespace(free=10*MIB))
    with pytest.raises(budget.StorageBudgetError, match="free-space reserve"):
        with handle.writing(4096): pytest.fail("disk floor must reject before IO")
    assert state(path)["claims"][str(handle.root)]["charged_bytes"] == 0


def test_no_tree_scan_per_shard_and_symlink_swap_still_refuses(tmp_path, monkeypatch):
    path = policy_file(tmp_path); handle = reserve(path)
    original = budget._inventory
    monkeypatch.setattr(budget, "_inventory", lambda root: pytest.fail("per-shard cache tree scan") if root==handle.root else original(root))
    with handle.writing(4096): pass
    handle.root.parent.mkdir(); handle.root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(budget.StorageBudgetError, match="symlink"):
        with handle.writing(4096): pytest.fail("retargeted root")


def test_legacy_reservation_is_immutable_and_counts_against_cap(tmp_path):
    evidence = tmp_path / "legacy_plan.json"; evidence.write_text('{"immutable_plan": true}')
    legacy = {"claim_id": "old-runtime", "root": str(tmp_path / "cache" / "old"),
              "evidence_path": str(evidence), "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
              "reserved_bytes": 4*MIB}
    path = policy_file(tmp_path, cap=7*MIB, legacy=[legacy])
    initialized = budget.StorageBudget(path).initialize()
    assert initialized['legacy_reserved_bytes'] == 4*MIB and initialized['guarded_claims'] == 0
    assert initialized['campaign_reserved_bytes'] == 4*MIB+LEDGER
    handle = reserve(path)
    assert handle.receipt["campaign_reserved_bytes"] == 7*MIB+LEDGER
    value = json.loads(path.read_text()); value["claims"].append(claim(tmp_path / "cache" / "b", 1))
    write_json(path, value)
    with pytest.raises(budget.StorageBudgetError, match="aggregate"):
        reserve(path, value["claims"][1])
    evidence.write_text("changed")
    with pytest.raises(budget.StorageBudgetError, match="evidence changed"):
        reserve(path)


def test_interrupted_ledger_metadata_has_a_hard_allowance(tmp_path):
    path=policy_file(tmp_path); handle=reserve(path)
    (tmp_path/'ledger'/'CLAIMS.json.interrupted.tmp').write_bytes(b'x'*LEDGER)
    with pytest.raises(budget.StorageBudgetError, match='ledger metadata/staging'):
        with handle.writing(1): pytest.fail('unbounded ledger leftovers')
    assert (tmp_path/'ledger'/'CLAIMS.json.interrupted.tmp').exists()


def teacher_fixture(tmp_path, monkeypatch, command="build"):
    import torch
    from PIL import Image
    from starVLA.dataloader.st_world_history import image_reference
    from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
    from tools.st_world_policy import cache_teacher
    image = tmp_path / "image.png"; Image.new("RGB",(32,18),"blue").save(image)
    contract = TeacherFeatureContract(image_hw=(32,32),patch_hw=(16,16),native_grid_hw=(2,2),output_grid_hw=(2,2),
        channels=8,storage_dtype="float32",compute_dtype="float32",view_names=["CAM_F0","CAM_L0","CAM_R0"],
        teacher_identity="a"*64,transform_identity="b"*64,pooling={"kind":"none","kernel_hw":[1,1],"stride_hw":[1,1]})
    manifest = {"schema":"st_teacher_requests_v1","split":"train","times_s":[0.],
        "view_names":list(contract.view_names),"timestamp_tolerance_s":.05,"scenes":[
            {"token":"a","log":"l","split":"train","timestamp_us":0,"frames":[
                {"requested_s":0.,"timestamp_us":0,"views":[image_reference(image)]*3}]}]}
    manifest["identity"] = identity_hash(manifest)
    requests = tmp_path / "requests.json"; write_json(requests,manifest)
    registration = tmp_path / "registration.json"
    write_json(registration,{"phase":"teacher_cache","request_manifest_identity":manifest["identity"],
        "max_scenes":1,"storage_dtype":"float32","variant":"SP2"})
    args = cache_teacher.parser().parse_args([command,"--requests",str(requests),"--teacher-root",str(tmp_path),
        "--output",str(tmp_path/"cache"/"a"),"--variant","SP2","--storage-dtype","float32","--max-scenes","1",
        "--mode","registered","--registration",str(registration),"--device","cpu","--min-free-gib","0"])
    monkeypatch.setattr(cache_teacher,"make_contract",lambda *a,**k:(contract,{},{}))
    class FakeTeacher(torch.nn.Module):
        patch_size=(16,16); feature_dim=8; prefix=5
        def __init__(self,*a,**k):
            super().__init__(); self.eval()
            assert (tmp_path/"ledger"/"CLAIMS.json").exists(), "model preceded reservation"
        def forward(self,pixels): return torch.zeros(len(pixels),8,2,2)
    monkeypatch.setattr(cache_teacher.dino,"DINOv3TargetEncoder",FakeTeacher)
    return cache_teacher,args,contract,manifest


@pytest.mark.parametrize('command',['build','derive-pooled'])
def test_registered_builder_requires_policy_before_teacher_or_output(tmp_path, monkeypatch, command):
    teacher,args,_,_ = teacher_fixture(tmp_path,monkeypatch,command)
    monkeypatch.setattr(teacher,"make_contract",lambda *a,**k:pytest.fail("descriptor preceded policy requirement"))
    with pytest.raises(ValueError,match="require --storage-budget"):
        teacher.run(args)
    assert not Path(args.output).exists()


def test_registered_teacher_claim_plan_build_resume_and_verify(tmp_path, monkeypatch):
    teacher,args,contract,_ = teacher_fixture(tmp_path,monkeypatch,"plan")
    plan = teacher.run(args)
    row = claim(Path(args.output)); row.update(plan["storage_requirements"])
    row["cache_identity"] = plan["cache_identity"]
    args.storage_budget = str(policy_file(tmp_path,[row])); args.command="build"
    done = teacher.run(args)
    assert done["status"] == "COMPLETE" and done["identity"] == plan["cache_identity"]
    recorded = json.loads((Path(args.output)/"identity.json").read_text())
    assert recorded["execution"]["writer_sha256"] == hashlib.sha256(Path(teacher.__file__).read_bytes()).hexdigest()
    assert "storage_budget" not in recorded["execution"] and recorded["contract"] == contract.to_dict()
    assert teacher.run(args)["identity"] == done["identity"]
    args.command="finalize"
    assert teacher.run(args)["identity"] == done["identity"]
    args.command="verify"; args.storage_budget=None
    assert teacher.run(args)["status"] == "VERIFIED"
    assert len(state(Path(tmp_path/"policy.json"))["claims"]) == 1


def test_resource_policy_does_not_change_plan_namespace_or_contract(tmp_path, monkeypatch):
    teacher,args,_,_ = teacher_fixture(tmp_path,monkeypatch,"plan")
    args.storage_budget="/not-read-by-plan/policy-a.json"; a=teacher.run(args)
    args.storage_budget="/not-read-by-plan/policy-b.json"; b=teacher.run(args)
    assert a["cache_identity"] == b["cache_identity"]
    assert a["contract_identity"] == b["contract_identity"]


def test_guarded_teacher_receipt_crash_resumes_and_preserves_orphan(tmp_path, monkeypatch):
    from starVLA.dataloader import st_world_targets as target_io
    teacher,args,_,_=teacher_fixture(tmp_path,monkeypatch,'plan')
    plan=teacher.run(args); row=claim(Path(args.output));row.update(plan['storage_requirements'])
    row['cache_identity']=plan['cache_identity'];args.storage_budget=str(policy_file(tmp_path,[row]))
    args.command='build'; original=target_io.atomic_json
    def crash(path,*a,**k):
        if Path(path).name=='shard_000000.json':raise RuntimeError('injected receipt failure')
        return original(path,*a,**k)
    monkeypatch.setattr(target_io,'atomic_json',crash)
    with pytest.raises(RuntimeError,match='injected receipt'):teacher.run(args)
    assert (Path(args.output)/'shard_000000.safetensors').exists()
    assert not (Path(args.output)/'COMPLETE.json').exists()
    monkeypatch.setattr(target_io,'atomic_json',original)
    assert teacher.run(args)['status']=='COMPLETE'
    assert len(list(Path(args.output).glob('*.orphan_*.safetensors')))==1
    assert len(state(Path(args.storage_budget))['claims'])==1


def test_registered_pooled_derivation_uses_guard_and_no_encoder(tmp_path, monkeypatch):
    import torch
    from starVLA.dataloader.st_world_schema import TeacherFeatureContract
    from starVLA.dataloader.st_world_targets import TeacherCacheWriter,TeacherCacheReader
    teacher,args,contract,manifest=teacher_fixture(tmp_path,monkeypatch,'plan')
    value=contract.to_dict();value.update(image_hw=[288,384],native_grid_hw=[18,24],output_grid_hw=[18,24])
    native=TeacherFeatureContract.from_dict(value)
    source=TeacherCacheWriter(tmp_path/'source',native,manifest,execution={'teacher':{},'transform':{}})
    features=torch.arange(18*24*8,dtype=torch.float32).reshape(1,18,24,8)
    source.commit_shard(0,features,torch.ones(1,18,24,dtype=torch.bool));source.finalize()
    value.update(output_grid_hw=[6,8],pooling={'kind':'avg','kernel_hw':[3,3],'stride_hw':[3,3]})
    monkeypatch.setattr(teacher,'make_contract',lambda *a,**k:(TeacherFeatureContract.from_dict(value),{},{}))
    monkeypatch.setattr(teacher.dino,'DINOv3TargetEncoder',lambda *a,**k:pytest.fail('derived cache loaded encoder'))
    args.variant='SP1';args.source_cache=str(source.root);args.source_cache_identity=source.identity['identity']
    registration=json.loads(Path(args.registration).read_text());registration['variant']='SP1';write_json(Path(args.registration),registration)
    plan=teacher.run(args);row=claim(Path(args.output));row.update(plan['storage_requirements']);row['cache_identity']=plan['cache_identity']
    args.storage_budget=str(policy_file(tmp_path,[row]));args.command='derive-pooled'
    done=teacher.run(args)
    assert done['status']=='COMPLETE' and done['teacher_forward_calls']==0
    reader=TeacherCacheReader(args.output,expected_identity=done['identity'])
    expected=torch.nn.functional.avg_pool2d(features.permute(0,3,1,2),3,3).permute(0,2,3,1)
    assert torch.equal(reader.read_scene('a')['current_dense_features'][0],expected[0])
    assert len(state(Path(args.storage_budget))['claims'])==1


def test_legacy_cpu_finalization_keeps_original_writer_identity(tmp_path, monkeypatch):
    import torch
    from starVLA.dataloader.st_world_targets import TeacherCacheWriter
    teacher,args,contract,manifest = teacher_fixture(tmp_path,monkeypatch,"finalize")
    execution={"teacher":{},"transform":{},"batch_size":args.batch_size,"writer_sha256":"e"*64,
               "format":"safetensors_uncompressed","optimizer_updates":0}
    writer=TeacherCacheWriter(args.output,contract,manifest,images_per_shard=args.images_per_shard,execution=execution)
    writer.commit_shard(0,torch.zeros(1,2,2,8),torch.ones(1,2,2,dtype=torch.bool))
    monkeypatch.setattr(teacher.dino,"DINOv3TargetEncoder",lambda *a,**k:pytest.fail("CPU finalization loaded teacher"))
    result=teacher.run(args)
    assert result["status"]=="COMPLETE" and result["identity"]==writer.identity["identity"]
    assert json.loads((Path(args.output)/"identity.json").read_text())["execution"]==execution
