"""Read committed RL journals against the unchanged source epoch sampler.

No model inference, scoring, sampling, optimizer step or trainer mutation. This
separates scene revisits and source even-batch padding from fresh G8 chains.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint
from starVLA.rl.simwam_recipe.data_stream import SceneStream
from starVLA.rl.simwam_recipe.invocation import validate_rows
from starVLA.rl.simwam_recipe.recovery_state import (
    rollout_digest, validate_boundary, validate_ranks,
)


def read_prefix(path, updates):
    """Hash exactly the committed prefix, excluding a concurrently growing tail."""
    rows = []
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                break
            rows.append(json.loads(line))
            digest.update(line)
            if len(rows) == updates:
                break
    validate_rows(rows, updates)
    return rows, digest.hexdigest()


def analyze_rows(ranks, stream, run_identity, updates):
    if updates <= 0 or updates % 4 or len(ranks) != stream.world:
        raise ValueError('Require complete four-inner, all-rank collections')
    for rows in ranks:
        validate_rows(rows, updates)
    seen, chains, epochs = Counter(), set(), {}
    padding = 0
    probes = {1, updates // 4}
    for c in range(1, updates // 4 + 1):
        cursor = (c - 1) * stream.world
        group_rows = []
        for rank, rows in enumerate(ranks):
            block = rows[(c - 1) * 4:c * 4]
            first = block[0]
            token, position = stream.at(cursor, rank)
            if first['token'] != token:
                raise ValueError(f'Source permutation mismatch collection={c} rank={rank}')
            for row in block:
                if (row['run'] != run_identity or row['behavior']['run'] != run_identity or
                        row['raw_scene_exposures'] != c * stream.world or
                        row['new_candidates'] != c * stream.world * 8 or
                        any(row[k] != first[k] for k in
                            ('token', 'chain_identity', 'behavior'))):
                    raise ValueError('Changed scene, chain, old probability or budget within inner reuse')
            chain = first['chain_identity']['chain']
            if chain in chains:
                raise ValueError('Repeated saved-chain digest is not fresh collection evidence')
            chains.add(chain)
            seen[token] += 1
            padding += int(position['padding'])
            e = epochs.setdefault(position['epoch'], dict(
                scene_exposures=0, padding_exposures=0, unique_tokens=set()))
            e['scene_exposures'] += 1
            e['padding_exposures'] += int(position['padding'])
            e['unique_tokens'].add(token)
            if position['padding'] or position['index'] == 0:
                probes.update(x for x in (c - 1, c, c + 1) if 1 <= x <= updates // 4)
            group_rows.append(first)
        if any(row['behavior'] != group_rows[0]['behavior'] for row in group_rows):
            raise ValueError('Different behavior policy versions within a global collection')
    exposure = updates // 4 * stream.world
    epoch_rows = [dict(epoch=k, scene_exposures=v['scene_exposures'],
        padding_exposures=v['padding_exposures'], unique_scene_count=len(v['unique_tokens']))
        for k, v in sorted(epochs.items())]
    result = dict(optimizer_updates=updates, fresh_global_rollouts=updates // 4,
        scene_exposures=exposure, fresh_candidates=exposure * 8,
        hard_population_scenes=len(stream.tokens), padded_epoch_size=stream.epoch_size,
        unique_scene_count=len(seen), unique_log_count=len({stream.logs[t] for t in seen}),
        scene_revisits_after_first_exposure=exposure-len(seen),
        epoch_padding_exposures=padding,
        nonpadding_revisits_after_first_exposure=exposure-len(seen)-padding,
        exposure_frequency_histogram={str(k): v for k, v in sorted(Counter(seen.values()).items())},
        distinct_chain_digests_in_journal=len(chains), epochs=epoch_rows)
    return result, sorted(probes)


def verify_checkpoint(run, identity, updates):
    ck = run / 'checkpoints' / f'update{updates:06d}_inner0'
    seal = json.loads((ck / 'COMPLETE').read_text())
    files = json.loads((ck / 'files.json').read_text())
    phase = 'COMPLETE' if updates == identity['recipe']['max_steps'] else 'READY'
    required = {'identity.json', 'transaction.json', 'optimizer.pt',
                *[f'rank{r}.pt' for r in range(identity['world'])]}
    if (seal['phase'] != phase or seal['updates'] != updates or seal['next_inner'] != 0 or
            seal['world'] != identity['world'] or not required <= files.keys() or
            file_sha(ck / 'files.json') != seal['files_sha256']):
        raise ValueError('Unsealed or wrong checkpoint boundary')
    for name, sha in files.items():
        if Path(name).name != name or file_sha(ck / name) != sha:
            raise ValueError('Changed checkpoint payload: ' + name)
    if json.loads((ck / 'identity.json').read_text()) != identity:
        raise ValueError('Different checkpoint identity')
    transaction = json.loads((ck / 'transaction.json').read_text())
    expected = dict(phase=phase, updates=updates, collections=updates // 4,
        scene_cursor=updates // 4 * identity['world'], next_inner=0,
        run=identity['sha256'], pending=False)
    if (validate_ranks(transaction['ranks']) != [expected] * identity['world'] or
            transaction['execution_identity'] != identity['execution_identity']):
        raise ValueError('Different transaction ancestry')
    for rank in range(identity['world']):
        saved = torch.load(ck / f'rank{rank}.pt', map_location='cpu', weights_only=False)
        if (saved['identity_sha256'] != identity['sha256'] or
                validate_boundary(saved['state'], saved['rollout'], identity) != expected):
            raise ValueError('Different rank boundary')
    return dict(path=str(ck), files_sha256=seal['files_sha256'], phase=phase)


def verify_behaviors(run, ranks, stream, collections):
    evidence = []
    for c in collections:
        for rank, rows in enumerate(ranks):
            row = rows[(c - 1) * 4]
            path = run / 'behavior' / f'collection{c:06d}' / f'rank{rank}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            token, position = stream.at((c - 1) * stream.world, rank)
            digest = rollout_digest(saved)
            if (saved['token'] != token or saved['log'] != stream.logs[token] or
                    saved['stream_position'] != position or
                    saved['behavior_version'] != row['behavior'] or
                    digest != row['chain_identity'] or digest != saved['chain_identity']):
                raise ValueError(f'Behavior payload mismatch collection={c} rank={rank}')
            evidence.append(dict(collection=c, rank=rank, stream_position=position,
                behavior_update=saved['behavior_version']['update'],
                behavior_invocation=saved['invocation'],
                first_inner_invocation=row['invocation'],
                token_sha256=fingerprint(token), payload_sha256=file_sha(path),
                chain_identity=digest))
    return evidence


def main():
    started, cpu_started = time.monotonic(), time.process_time()
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--updates', required=True, type=int)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run, output = Path(a.run), Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    identity = json.loads((run / 'identity.json').read_text())
    body = dict(identity); body.pop('sha256')
    if (fingerprint(body) != identity['sha256'] or identity['purpose'] != 'formal' or
            identity['world'] != 8 or identity['recipe']['variant'] != 'R_SOURCE'):
        raise ValueError('Not an authenticated formal eight-rank transfer run')
    data = identity['data_stream_identity']
    if (data['schema'] != 'fm_transfer_fixed_new_SFT_hard_population_v1' or
            data['original_source_epoch_padding'] is not True or
            file_sha(a.manifest) != data['manifest_sha256'] or
            file_sha(a.manifest) != identity['hard_manifest_sha256']):
        raise ValueError('Changed hard population or sampler contract')
    root = Path(__file__).resolve().parents[3]
    dependencies = ['contracts.py', 'data_stream.py', 'vendor/samplers.py',
                    'invocation.py', 'recovery_state.py']
    dependencies = ['starVLA/rl/simwam_recipe/' + name for name in dependencies]
    sources = {name: file_sha(root / name) for name in dependencies}
    if any(identity['source_files'][name] != sha for name, sha in sources.items()):
        raise ValueError('Analysis must reuse the actual frozen source sampler and validators')
    stream = SceneStream(a.manifest, identity['world'])
    if (len(stream.tokens) != data['scene_count'] or
            stream.manifest['screen_contract'] != data['screen_contract']):
        raise ValueError('Different SFT screen population')
    checkpoint = verify_checkpoint(run, identity, a.updates)
    parsed = [read_prefix(run / f'metrics_rank{rank}.jsonl', a.updates) for rank in range(8)]
    ranks = [x[0] for x in parsed]
    summary, collections = analyze_rows(ranks, stream, identity['sha256'], a.updates)
    evidence = verify_behaviors(run, ranks, stream, collections)
    result = dict(schema='fm_transfer_committed_RL_population_v1',
        status='COMPLETE_READONLY_POPULATION_AUDIT_NOT_POLICY_EVALUATION',
        captured_utc=datetime.now(timezone.utc).isoformat(),
        analysis_identity=file_sha(__file__), run_identity=identity['sha256'],
        checkpoint=checkpoint, manifest_sha256=stream.sha, reused_source_files=sources,
        journal_prefix_sha256=[x[1] for x in parsed], population=summary,
        checked_behavior_payloads=evidence,
        analysis_new_candidates=0, analysis_optimizer_updates=0, analysis_scores=0,
        analysis_main_wall_seconds=time.monotonic()-started,
        analysis_main_process_cpu_seconds=time.process_time()-cpu_started,
        limits=[
            'Counts describe only the effective committed journals; failed attempts remain in the separate cost ledger.',
            'All journal scene/chain identities are checked; tensor payload hashes are checked at the listed boundary probes.',
            'Behavior and first-inner invocation IDs are separately retained: a legal restore may span invocations; this does not replace the resume ledger.',
            'Padding and ordinary scene revisits generate fresh current-behavior G8 chains, not additional unique scenes.',
            'Four inner updates reuse a single fresh collection; this is not a new score or model-selection milestone.',
            'Analysis timers exclude Python imports and final JSON publication; no GPU is used.'])
    atomic_json(output, result)
    print(json.dumps(dict(output=str(output), population=summary,
                         checked_behavior_payloads=len(evidence)), ensure_ascii=False))


if __name__ == '__main__':
    main()
