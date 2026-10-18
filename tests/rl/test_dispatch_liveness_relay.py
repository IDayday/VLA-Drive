import hashlib
import json

import pytest

from scripts.analysis.fm_transfer.dispatch_liveness_relay import checked_live, digest, load_proof


def fixture_proof(tmp_path):
    queue = tmp_path / 'queue'
    p = queue / 'workers' / 'model' / 'receipt.json'
    p.parent.mkdir(parents=True)
    record = dict(host='remote', pid=42, start_ticks=123, state='STARTED', sha256='receipt')
    p.write_text(json.dumps(record))
    proof = dict(schema='final_dispatch_pid_relay_v1', queue=str(queue), started_unix=995,
                 finished_unix=1000, receipt_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest()},
                 checks={'receipt': dict(record=record, alive=True)})
    proof['sha256'] = digest(proof)
    path = tmp_path / 'proof.json'
    path.write_text(json.dumps(proof))
    return path, queue, p, record


def test_fresh_inventory_and_local_recheck(tmp_path):
    path, queue, _, record = fixture_proof(tmp_path)
    proof = load_proof(path, queue, now=1010)
    assert checked_live(proof, record, lambda _: False, host='coordinator')
    assert not checked_live(proof, record, lambda _: False, host='remote')


def test_reject_stale_or_changed_queue(tmp_path):
    path, queue, p, _ = fixture_proof(tmp_path)
    with pytest.raises(ValueError, match='stale'):
        load_proof(path, queue, now=1121)
    p.write_text('{}')
    with pytest.raises(ValueError, match='Queue changed'):
        load_proof(path, queue, now=1010)


def test_receipt_and_seal_cannot_be_substituted(tmp_path):
    path, queue, _, record = fixture_proof(tmp_path)
    proof = load_proof(path, queue, now=1010)
    with pytest.raises(ValueError, match='exact dispatch receipt'):
        checked_live(proof, dict(record, pid=43), lambda _: True)
    proof['checks']['receipt']['alive'] = False
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='unsealed'):
        load_proof(path, queue, now=1010)


def test_make_proof_checks_actual_receipts_and_never_overwrites(tmp_path):
    from types import SimpleNamespace
    from scripts.analysis.fm_transfer.dispatch_liveness_relay import make_proof
    _,queue,_,record=fixture_proof(tmp_path);called=[]
    dispatch=SimpleNamespace(__file__=__file__,authenticated=lambda p:json.loads(p.read_text()),
        live=lambda r:called.append(r) or True)
    path=tmp_path/'proofs/fresh.json';proof=make_proof(queue,dispatch,path)
    assert called==[record]
    assert load_proof(path,queue)['checks'][record['sha256']]['alive'] is True
    with pytest.raises(FileExistsError):make_proof(queue,dispatch,path)


def test_make_proof_does_not_translate_ssh_failure_into_dead_worker(tmp_path):
    from types import SimpleNamespace
    from scripts.analysis.fm_transfer.dispatch_liveness_relay import make_proof
    _,queue,_,_=fixture_proof(tmp_path)
    def unavailable(_):raise RuntimeError('SSH unavailable')
    dispatch=SimpleNamespace(__file__=__file__,authenticated=lambda p:json.loads(p.read_text()),live=unavailable)
    path=tmp_path/'failed.json'
    with pytest.raises(RuntimeError,match='SSH unavailable'):make_proof(queue,dispatch,path)
    assert not path.exists()
