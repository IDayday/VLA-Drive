"""CPU process fixtures only; no data, canonical scoring or real admission."""
import json,multiprocessing,os,resource,signal,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tools.st_world_policy import run_road_direction as r

def fixture_worker(*args):
    if os.environ['ROAD_FIXTURE_MODE']=='native_exit':os._exit(23)
    time.sleep(60)

def scenario(mode):
    owned=[];context=multiprocessing.get_context('spawn')
    class Context:
        def Process(self,*a,**kw):
            p=context.Process(*a,**kw);owned.append(p);return p
    actual_context=multiprocessing.get_context;actual_child=r._child_entry
    actual_check=r.check_admission;actual_glob=Path.rglob
    with tempfile.TemporaryDirectory(prefix='road_systems_fixture_') as tmp:
        p=Path(tmp);out=p/'out';ep=p/'evidence.json'
        evidence={'selected':[{'scene_id':f'fixture{i}'} for i in range(32)],'selection_sha256_before_scores':'fixture'}
        ep.write_text(json.dumps(evidence));rp=p/'registration.json';rp.write_text('{}')
        reg={'output_root':str(out),'source_hashes':{'fixture':'only'},'evidence':{'path':str(ep),'sha256':r.sha(ep.read_bytes())}}
        r.check_admission=lambda *args:(reg,{'attempt_id':mode})
        r._child_entry=fixture_worker
        multiprocessing.get_context=lambda kind:Context()
        if mode=='monitor_error':
            def broken_glob(self,*args,**kwargs):
                if self==out:raise FileNotFoundError('injected vanished atomic temporary file')
                return actual_glob(self,*args,**kwargs)
            Path.rglob=broken_glob
        os.environ['ROAD_FIXTURE_MODE']=mode
        start=time.monotonic();error=None;code=None
        try:code=r.supervise(rp,p/'unused.json','fixture',False)
        except BaseException as e:error=type(e).__name__+': '+str(e)
        finally:
            live_before=[c.is_alive() for c in owned]
            for child in owned:
                if child.is_alive():child.terminate();child.join(timeout=3)
                if child.is_alive():child.kill();child.join(timeout=3)
            Path.rglob=actual_glob;multiprocessing.get_context=actual_context
            r._child_entry=actual_child;r.check_admission=actual_check
        resultpath=out/('RESULT_'+mode+'.json')
        result=json.loads(resultpath.read_text()) if resultpath.exists() else None
        return {'mode':mode,'return_code':code,'error':error,'child_alive_after_supervise':live_before,
                'fixture_cleanup_remaining':sum(c.is_alive() for c in owned),'summary_exists':result is not None,
                'summary_complete_scenes':result.get('complete_scenes') if result else None,
                'summary_requested_scenes':result.get('requested_scenes') if result else None,
                'child_exitcodes':[c.exitcode for c in owned],'wall_seconds':time.monotonic()-start}

if __name__=='__main__':
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
    rows=[scenario('native_exit'),scenario('monitor_error')]
    after=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'target_author_commit':'55721af755637636b1e769a17e175dabd8bcc033','integrated_target':'a7d0a01583027dca3b04a84b5ab51d3fb2989984',
            'status':'FAIL_MONITOR_EXCEPTION_ORPHANS_WORKER','rows':rows,'official_calls':0,'real_asset_reads':0,'gpu_calls':0,
            'wall_seconds':time.monotonic()-started,'parent_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            'child_cpu_seconds':children.ru_utime+children.ru_stime,'peak_rss_parent_kib':after.ru_maxrss,'peak_rss_child_kib':children.ru_maxrss}
    (Path(__file__).resolve().parent/'RUNNER_FAULTS_ATTEMPT1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
