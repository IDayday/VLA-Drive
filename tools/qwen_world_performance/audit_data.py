"""Full registered population file census; no future data is exposed to actors."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json,lzma,pickle,time,os
from pathlib import Path
from starVLA.rl.qwen_world_performance.data_protocol import (
    DEFAULT_SOURCE,source_population,write_manifest,atomic_json,file_sha,physical,sealed)


def audit(source,output,workers=8):
    started=time.time();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    m,data,pop,logs=source_population(source);root=Path(m['data_root'])
    complete=json.loads((root/'COMPLETE').read_text())
    if complete['manifest_sha256']!=file_sha(root/'manifest.json')or file_sha(root/'records.pkl')!=data['records_sha256']:
        raise ValueError('Observation seal/records SHA mismatch')
    with(root/'records.pkl').open('rb')as f:records=pickle.load(f)
    if set(records)!=set(m['tokens']):raise ValueError('Record population mismatch')
    roots=[Path(p)for p in data['canonical_metric_cache_roots']]
    def one(t):
        row=records[t];errors=[];images=[];caches=[]
        if row.get('token')!=t or row.get('log')!=logs[t]:errors.append(dict(kind='record_identity',token=t))
        for cam in ('cam_l0','cam_f0','cam_r0'):
            raw=row['images'].get(cam);p=Path(raw)if raw else None
            try:
                st=p.stat()if p else None
                if st is None or not p.is_file()or st.st_size==0:raise FileNotFoundError(str(p))
                images.append((cam,str(p),st.st_size))
            except(OSError,ValueError)as e:errors.append(dict(kind='missing_or_empty_current_image',token=t,camera=cam,path=raw,error=repr(e)))
        for base in roots:
            p=base/logs[t]/'unknown'/t/'metric_cache.pkl'
            try:st=p.stat()
            except FileNotFoundError:continue
            except OSError as e:errors.append(dict(kind='cache_stat_error',token=t,path=str(p),error=repr(e)));continue
            issue=None
            try:
                with p.open('rb')as f:head=f.read(16);f.seek(max(0,st.st_size-12));tail=f.read()
                if st.st_size<32 or head[:6]!=b'\xfd7zXZ\x00'or tail[-2:]!=b'YZ':issue='invalid/truncated XZ framing'
                else:
                    # Bound decoded bytes; this is not a full payload decode/hash.
                    with p.open('rb')as f:prefix=lzma.LZMADecompressor().decompress(f.read(min(4096,st.st_size)),max_length=8)
                    if not prefix.startswith(b'\x80'):issue='XZ payload does not begin with a binary pickle protocol'
            except(OSError,lzma.LZMAError,ValueError)as e:issue=repr(e)
            caches.append(dict(root=str(base),path=str(p),bytes=st.st_size,inode=st.st_ino,device=st.st_dev,framing_issue=issue))
            if issue:errors.append(dict(kind='cache_framing_or_prefix_error',token=t,path=str(p),error=issue))
        if not caches:errors.append(dict(kind='missing_metric_cache',token=t,log=logs[t]))
        return t,images,caches,errors
    cache_counts=Counter();duplicates=[];errors=[];ledger=[];observation_keys=defaultdict(list);inode_tokens=defaultdict(set)
    with ThreadPoolExecutor(max_workers=workers)as pool:
        for t,images,caches,bad in pool.map(one,m['tokens']):
            errors.extend(bad);cache_counts.update(c['root']for c in caches)
            if len(caches)>1:duplicates.append(dict(token=t,paths=[c['path']for c in caches],same_inode=len({(c['device'],c['inode'])for c in caches})==1))
            for c in caches:inode_tokens[(c['device'],c['inode'])].add(t)
            row=records[t];observation_keys[(physical(logs[t]),str(row.get('timestamp')),tuple(x[1]for x in images))].append(t)
            ledger.append(dict(token=t,log=logs[t],physical_log=physical(logs[t]),current_images_checked=len(images),cache_count=len(caches),selected_cache=caches[0]['path']if caches else None,selected_cache_bytes=caches[0]['bytes']if caches else None,issues=len(bad)))
    out_tokens=set(m['tokens']);test=set(pop['navtest_tokens']);trainlogs={logs[t]for t in out_tokens};testlogs={pop['token_logs'][t]for t in test}
    checks=dict(registered_count=len(m['tokens']),unique_tokens=len(out_tokens),token_duplicates=len(m['tokens'])-len(out_tokens),records_count=len(records),NAVTEST_count=len(test),token_overlap=sorted(out_tokens&test),segment_log_overlap=sorted(trainlogs&testlogs),physical_log_overlap=sorted({physical(x)for x in trainlogs}&{physical(x)for x in testlogs}),TRAIN_segment_logs=len(trainlogs),TRAIN_physical_logs=len({physical(x)for x in trainlogs}),NAVTEST_physical_logs=len({physical(x)for x in testlogs}),old_DEV_train_diagnostic_count=len(pop['dev_tokens']),current_image_files_checked=sum(r['current_images_checked']for r in ledger),cache_files_framing_checked=sum(r['cache_count']for r in ledger),cache_root_counts=dict(cache_counts),cache_missing_scenes=sum(r['cache_count']==0 for r in ledger),error_counts=dict(Counter(e['kind']for e in errors)),duplicate_cache_scenes=len(duplicates),duplicate_current_observation_groups=[v for v in observation_keys.values()if len(v)>1],cache_inode_shared_between_distinct_tokens=[sorted(v)for v in inode_tokens.values()if len(v)>1])
    for name,rows in [('FILE_LEDGER.jsonl',ledger),('BROKEN_OR_MISSING.jsonl',errors),('DUPLICATE_CACHES.jsonl',duplicates)]:
        with(output/name).open('w')as f:
            for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    manifest=write_manifest(output/'NAVTRAIN_SCENES.json',source)
    report=dict(schema='world_performance_full_NAVTRAIN_asset_audit_v1',status='COMPLETE'if not errors else'COMPLETE_WITH_ASSET_ISSUES',created_utc=datetime.now(timezone.utc).isoformat(),duration_seconds=time.time()-started,checks=checks,source_manifest=str(Path(source).resolve()),source_manifest_sha256=file_sha(source),records_sha256=data['records_sha256'],manifest=str(output/'NAVTRAIN_SCENES.json'),manifest_sha256=file_sha(output/'NAVTRAIN_SCENES.json'),file_ledger_sha256=file_sha(output/'FILE_LEDGER.jsonl'),errors_path=str(output/'BROKEN_OR_MISSING.jsonl'),duplicate_caches_path=str(output/'DUPLICATE_CACHES.jsonl'),content_validation='ALL cache files: existence/size/XZ magic+footer/bounded binary-pickle prefix only; not full decompression, unpickle, semantic metric validity or full-payload SHA. Current image files: stat/nonempty only; no pixel decode.',actor_contract='Only current L/F/R camera, ego, navigation. No requested_GT or privileged cache values enter sampling feedback/actor.',training_feedback='Only explicitly same-run TRAIN B1 feedback; NAVTEST scores/IDs cannot construct sampling queues.',old_DEV='Included in NAVTRAIN and TRAIN_DIAGNOSTIC, not independent heldout',no_scene_silently_excluded=True,no_new_score_or_model_forward=True)
    atomic_json(output/'AUDIT.json',sealed(report));return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',default=DEFAULT_SOURCE);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    r=audit(a.source,a.output,a.workers);print(json.dumps(dict(status=r['status'],checks={k:v for k,v in r['checks'].items()if not isinstance(v,list)},duration_seconds=r['duration_seconds'],audit=str(Path(a.output)/'AUDIT.json'))))
if __name__=='__main__':main()
