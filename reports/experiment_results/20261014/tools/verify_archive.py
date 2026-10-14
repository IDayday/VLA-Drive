"""Verify this archive offline: hashes, source blobs, score populations and paired statistics."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DATA = ROOT / 'data'
METRICS = ['PDMS', 'NC', 'DAC', 'TTC', 'EP', 'comfort', 'DDC']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open() as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    assert np.allclose(a, b, rtol=0, atol=1e-9), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true', help='Write a new manifest after successful checks; never edits old snapshots')
    parser.add_argument('--check-git', action='store_true', help='Also compare paths and scoped patches against locally available original Git commits')
    args = parser.parse_args()
    manifest_path = ROOT / 'MANIFEST.json'
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['MANIFEST.json', 'VERIFICATION.json'] and '__pycache__' not in p.parts)
    # Include the exporter's inner MANIFEST, which binds the captured live prefix.
    files += [DATA / 'MANIFEST.json']
    assert len(set(files)) == len(files)
    current = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}
    if not args.seal:
        assert current == read(manifest_path)['files'], 'Archive files changed or are missing'
    sources = read(ROOT/'evidence/SOURCE_INDEX.json')
    for item in sources:
        p = ROOT/'evidence'/item['file']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
    source_index = read(ROOT/'implementation/INDEX.json')
    versions = {v['version']: read(ROOT/'implementation'/v['manifest']) for v in source_index['versions']}
    source_count, source_hashes, patch_count = 0, set(), 0
    for version in source_index['versions']:
        v = versions[version['version']]
        assert v['commit'] == version['commit'] and len(v['files']) == version['files']
        if args.check_git:
            tree = subprocess.check_output(['git', 'ls-tree', '-r', v['commit']], cwd=REPO).decode()
            git_paths = {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2] for line in tree.splitlines()}
        for entry in v['files']:
            p = ROOT/'implementation'/entry['archive_path']
            raw = p.read_bytes()
            assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == entry['git_blob']
            if args.check_git:
                assert git_paths[entry['path']] == entry['git_blob']
            source_hashes.add(entry['sha256'])
            source_count += 1
        patch = v['comparison_patch']
        if patch:
            assert sha(ROOT/'implementation'/patch['path']) == patch['sha256']
            if args.check_git:
                previous = versions[patch['comparison_base']]
                assert previous['commit'] == patch['base_commit']
                paths = sorted({e['path'] for e in v['files']} | {e['path'] for e in previous['files']})
                original = subprocess.check_output(['git', 'diff', '--no-ext-diff', '--no-renames', '--full-index',
                                                    previous['commit'], v['commit'], '--', *paths], cwd=REPO)
                assert original == (ROOT/'implementation'/patch['path']).read_bytes()
            patch_count += 1
    assert len(source_hashes) == source_index['unique_files'] == 204

    # Numeric scene matrix with a common anonymized scene/log order.
    banks, population, ids, scene_index = {}, {}, [], {}
    current_key, records = None, []
    def finish(key, rr):
        nonlocal population, ids, scene_index
        assert len(rr) == 12146 and len({r['scene_id'] for r in rr}) == 12146
        member = {r['scene_id']: r['log_id'] for r in rr}
        assert len(set(member.values())) == 136
        if not population:
            population = member
            ids = sorted(member)
            scene_index = {s: i for i, s in enumerate(ids)}
        assert population == member
        values = np.empty((12146, len(METRICS)))
        for r in rr:
            assert r['status'] == 'ok'
            assert re.fullmatch('[a-f0-9]{24}', r['scene_id']) and re.fullmatch('[a-f0-9]{24}', r['log_id'])
            values[scene_index[r['scene_id']]] = [float(r[m]) for m in METRICS]
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 100
        assert key not in banks
        banks[key] = values
    with gzip.open(DATA/'G_NAVTEST_SCENES.csv.gz','rt') as stream:
        for r in csv.DictReader(stream):
            key = r['evaluation_id']
            if current_key is not None and key != current_key:
                finish(current_key, records)
                records = []
            current_key = key
            records.append(r)
    finish(current_key, records)
    assert len(banks) == 42
    nav = rows(DATA/'NAVSIM_NAVTEST.csv')
    for r in nav:
        key = f"navtest:G:{r['group'][:2]}:{r['updates']}:{r['variant']}:42"
        values = banks[key]
        close(values.mean(0), [float(r[m]) for m in METRICS])
        assert int(r['zero_scenes']) == int((values[:,0] == 0).sum())
        assert int(r['failed']) == 0 and int(r['scenes']) == 12146 and int(r['logs']) == 136
    log_order = read(DATA/'G_BOOTSTRAP_LOG_ORDER.json')
    assert set(log_order) == set(population.values()) and len(log_order) == 136
    log_idx = {v:i for i,v in enumerate(log_order)}
    indexes = np.array([log_idx[population[s]] for s in ids])
    counts = np.bincount(indexes, minlength=136)
    draws = np.random.default_rng(42).integers(0,136,size=(20000,136))
    def paired_check(a, b, metrics, transfers, cross=False):
        for metric, stat in metrics.items():
            alias = {'Comfort':'comfort','Direction':'DDC'}.get(metric, metric)
            col = METRICS.index(alias)
            delta = a[:,col]-b[:,col]
            sums = np.bincount(indexes,weights=delta,minlength=136)
            bootstrap = sums[draws].sum(1)/counts[draws].sum(1)
            close(float(delta.mean()),stat['delta_points'])
            close(np.quantile(bootstrap,[.025,.975]),stat['paired_log_95CI'])
        az,bz=a[:,0]==0,b[:,0]==0
        labels = ('fixed_baseline_zero','introduced_zero') if cross else ('fixed_baseline_zero_scenes','introduced_zero_scenes')
        assert int((bz & ~az).sum()) == transfers[labels[0]]
        assert int((~bz & az).sum()) == transfers[labels[1]]
    pairs = read(DATA/'PAIRED_COMPARISONS.json')['comparisons']
    for p in pairs:
        def key(x):
            return f'navtest:G:{x[0]}:{x[1]}:{x[2]}:42'
        paired_check(banks[key(p['candidate'])],banks[key(p['baseline'])],p['paired']['metrics'],p['paired']['zero_failure_transfers'])
    cross = read(DATA/'G2_HISTORICAL_COMPARISONS.json')['comparisons']
    needed = {p['baseline'] for p in cross}
    old = {k:np.full((12146,7),np.nan) for k in needed}
    seen = {k:set() for k in needed}
    with gzip.open(ROOT.parent/'20261008/data/scenes/navtest.csv.gz','rt') as stream:
        for r in csv.DictReader(stream):
            k=r['evaluation_id']
            if k not in needed:
                continue
            assert r['scene_id'] not in seen[k] and population[r['scene_id']] == r['log_id'] and r['status']=='ok'
            seen[k].add(r['scene_id'])
            old[k][scene_index[r['scene_id']]]=[float(r.get({'comfort':'Comfort','DDC':'Direction'}.get(m,m))) for m in METRICS]
    for v in old.values():
        assert np.isfinite(v).all()
    for p in cross:
        paired_check(banks[p['candidate']],old[p['baseline']],p['metrics'],p,cross=True)

    sweep = {int(r['steps']):r for r in rows(ROOT/'evidence/fm_steps/RESULTS.csv')}
    sums, zeros, members = {}, {}, {}
    with gzip.open(DATA/'C1_FM_SWEEP_SCENES.csv.gz','rt') as stream:
        for r in csv.DictReader(stream):
            k=int(r['steps'])
            if k not in sums:
                sums[k]=np.zeros(7);zeros[k]=0;members[k]=set()
            assert r['scene_id'] not in members[k] and population[r['scene_id']]==r['log_id'] and r['status']=='ok'
            members[k].add(r['scene_id'])
            values=np.array([float(r[{'comfort':'Comfort','DDC':'Direction'}.get(m,m)]) for m in METRICS])
            sums[k]+=values;zeros[k]+=int(values[0]==0)
            if k==10:
                close(values,old['navtest:C:original:C1:100000:42'][scene_index[r['scene_id']]])
    for k in sweep:
        assert len(members[k])==12146
        close(sums[k][0]/12146,float(sweep[k]['PDMS']))
        assert zeros[k]==int(sweep[k]['zero_scenes']) and int(sweep[k]['failed'])==0

    all_rows=rows(DATA/'ALL_CHECKPOINTS.csv')
    assert len(all_rows)==202
    keys=[(r['method_id'],r['split'],r['update'],r['variant']) for r in all_rows]
    assert len(keys)==len(set(keys))
    for b in rows(DATA/'BEST_NAVTEST.csv'):
        rr=[r for r in all_rows if r['method_id']==b['method_id'] and r['split']=='navtest' and r['variant']!='q0']
        assert len(rr)==int(b['observed_checkpoints'])
        best=max(rr,key=lambda r:float(r['PDMS']))
        assert best['update']==b['best_update'] and best['PDMS']==b['PDMS']
    assert len(rows(DATA/'OBJECTIVE_MATURE_PILOTS.csv'))==16
    assert len(rows(DATA/'FROZEN_READOUT_PROBES.csv'))==8
    ledger=read(DATA/'METHOD_LEDGER.json')
    assert len(ledger)==67 and len({r['method_id'] for r in ledger})==67
    status={(r['dataset'],r['group'][:2]):r for r in rows(DATA/'TRAINING_STATUS.csv')}
    for name,step in [('G0',71052),('G1',84135)]:
        r=status[('navsim',name)];assert r['status']=='paused_by_user' and int(r['saved_update'])==step
    assert status[('nuscenes','G1')]['status']=='paused_by_user'
    assert int(status[('nuscenes','G1')]['saved_update'])==7811
    assert int(status[('navsim','G3')]['last_logged_update'])==100000
    assert int(status[('nuscenes','G3')]['last_logged_update'])==17424
    assert status[('navsim','G2')]['host']=='training-vla-zt2'
    assert int(status[('navsim','G2')]['last_logged_update'])==73775
    nu=rows(DATA/'NUSCENES_PLANNING.csv')
    assert len(nu)==20 and all(int(r['samples'])==4969 and int(r['scenes'])==150 for r in nu)

    # Check local links in new prose/catalogs; copied source docstrings are not new documentation.
    links=0
    for p in [*ROOT.glob('*.md'),*ROOT.glob('implementation/versions/*.md')]:
        for target in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)',p.read_text()):
            if target.startswith(('http://','https://','#','mailto:')):
                continue
            target=target.split('#',1)[0]
            destination = (p.parent/target).resolve()
            pending_seal = args.seal and destination in (manifest_path, ROOT/'VERIFICATION.json')
            assert destination.exists() or pending_seal, f'Broken link: {p.name}: {target}'
            links+=1
    result={'status':'PASS','checked_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),
            'new_archive_files':len(files),'original_evidence_receipts':len(sources),
            'implementation_versions':len(source_index['versions']),'version_file_mappings':source_count,
            'unique_exact_source_blobs':len(source_hashes),'scoped_patches':patch_count,
            'original_git_paths_and_patch_bytes_checked':args.check_git,
            'G_full_scene_banks':len(banks),'G_scene_rows':len(banks)*12146,
            'G_pairs_recomputed':len(pairs),'historical_pairs_recomputed':len(cross),
            'fixed_C1_inference_banks':len(sweep),'fixed_C1_scene_rows':len(sweep)*12146,
            'checkpoint_rows':len(all_rows),'local_markdown_links_checked':links,
            'aggregation_absolute_tolerance':1e-9,
            'scope':'Read-only archive/numeric/source verification; no model/GPU/checkpoint replay and no new inference.'}
    if args.seal:
        manifest={'schema':'c_to_g_incremental_archive_v1','sealed_UTC':result['checked_UTC'],
                  'base_report_commit':'ed0725c767a888403b6a6ef3490eb0fdfc929a23',
                  'branch':'analysis/c-to-g-experiment-results-20261014','files':current,
                  'report_commit':'The Git commit containing this manifest; not a training_source_sha',
                  'verification_file':'VERIFICATION.json'}
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        (ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
