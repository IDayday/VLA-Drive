"""Finite CPU completion publisher for one already-running frozen-head probe.

Does not train, dispatch, retry, signal models, or edit scientific runtimes.
Publishes only exact completed models to its unchanged isolated report branch.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--probe-root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--expected-head',required=True)
    p.add_argument('--timeout-seconds',type=int,default=3600);a=p.parse_args()
    root=a.probe_root.resolve();repo=a.repo.resolve();lock=(root/'publication.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=root/'PUBLICATION_STATE.json';started=time.time()
    write(state,dict(status='WAITING_COMPLETE',pid=os.getpid(),started=started,model_signals=0))
    while not(root/'outputs/COMPLETE.json').exists():
        launch=read(root/'LAUNCH.json');result=Path(launch['job'])/'RESULT.json'
        if result.exists()and read(result).get('returncode')!=0:
            write(state,dict(status='PROBE_FAILED_REVIEW_REQUIRED',result=str(result),model_signals=0));return
        if time.time()-started>a.timeout_seconds:
            write(state,dict(status='PUBLICATION_WAIT_TIMEOUT; PROBE_UNTOUCHED',model_signals=0));return
        time.sleep(60)
    complete=read(root/'outputs/COMPLETE.json');plan=read(root/'PLAN.json')
    if complete['status']!='PASS'or set(complete['results'])!=set('ABCD'):
        raise ValueError('Not a complete four-model diagnostic')
    for arch,result in complete['results'].items():
        if (result['status']!='PASS'or result['IL_update']!=plan['models'][arch]['updates']
                or result['optimizer_updates']!=0 or result['scorer_calls']!=0 or not result['parameters_unchanged']):
            raise ValueError('Frozen diagnostic contract mismatch')
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    if (git('branch','--show-current')!='perf/qwen-world-abcd-equivalent-20261031'
            or git('rev-parse','HEAD')!=a.expected_head or git('status','--porcelain')):
        write(state,dict(status='RESULTS_COMPLETE; PUBLICATION_NEEDS_REVIEW',reason='Report branch/HEAD/worktree changed',model_signals=0));return
    relative='reports/qwen_world_il_auxiliary_diagnosis_20261031';destination=repo/relative/'probe_results'
    for name in [*(arch+'_RESULT.json'for arch in'ABCD'),'COMPLETE.json']:
        shutil.copyfile(root/'outputs'/name,destination/name)
    lines=['# 四组冻结IL辅助任务诊断','',
        '共同TRAIN32；另64个TRAIN拟合位置均值，事实后置8场景。不同IL预算，不作架构排名。无更新/新评分。','',
        '| 组 | IL update | DINO相对均值误差降低 | WA前置 | WA事实后置 |',
        '|---|---:|---:|---:|---:|']
    for arch in'ABCD':
        result=complete['results'][arch];f=result['features']
        lines.append(f'| {arch} | {result["IL_update"]} | '+ ' | '.join(f'{100*f[k]["skill_vs_position_mean"]:.2f}%'for k in('dino','wa_pre','wa_post'))+' |')
    lines+=['','正值只证明这个TRAIN面板上优于固定位置均值；不能替代Navtest或证明辅助任务带来规划收益。',
            '前置32场景与后置8场景不直接比较；事件检出能力须结合各JSON的实际事件数与always-none基线。']
    (destination/'README.md').write_text('\n'.join(lines)+'\n')
    doc=repo/relative/'README.md';text=doc.read_text();text=text.replace('其他三组仍由同一个有限worker继续。','其余三组也已完成，见[四组冻结诊断](probe_results/README.md)。')
    doc.write_text(text)
    write(repo/relative/'PROBE_STATUS.json',dict(status='COMPLETE',job=read(root/'LAUNCH.json')['job'],
        plan=str(root/'PLAN.json'),model_signals=0,optimizer_updates=0,scorer_calls=0))
    subprocess.run(['git','add',relative],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Publish completed four-model frozen IL auxiliary diagnostics'],cwd=repo,check=True)
    sha=git('rev-parse','HEAD');subprocess.run(['git','push','origin','HEAD:refs/heads/perf/qwen-world-abcd-equivalent-20261031'],cwd=repo,check=True)
    remote=git('ls-remote','--heads','origin','refs/heads/perf/qwen-world-abcd-equivalent-20261031').split()[0]
    if remote!=sha:raise ValueError('Remote publication SHA mismatch')
    write(state,dict(status='PUSHED',commit=sha,remote_verified=True,model_signals=0,at=time.time()))


if __name__=='__main__':main()
