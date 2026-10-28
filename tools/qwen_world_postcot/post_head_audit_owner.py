"""Finite read-only post-head audit; preserve and give priority to Navtest jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from tools.qwen_world_postcot.latest_snapshot_owner import SnapshotRequest
from tools.qwen_world_postcot.user_interim_owner import read, write


def occupied_main(root):
    reserved = set(); launched = 0
    for path in root.glob('*/jobs/rank*/spec.json'):
        launched += 1; result = path.parent/'result.json'
        if not result.exists():
            reserved.update(read(path)['gpu_uuids'])
    return reserved, launched


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True)
    p.add_argument('--main-wave',required=True);p.add_argument('--report-dir',required=True)
    a=p.parse_args();root=Path(a.root);source=Path(__file__).resolve().parents[2]
    lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'owner.json').exists():raise FileExistsError('Never duplicate this finite audit')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    reg=read(root/'registration.json')
    write(root/'owner.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),source=str(source),source_sha=head))
    pin=dict(kind='IL',checkpoint=reg['checkpoint'],updates=reg['updates'],evaluation_hosts=[
        'training-vla-zt','training-vlawm-tsinghua-zt','training-vlawm-tsinghua-zt3',
        'training-vlawm-tsinghua-zt4','training-vlawm-tsinghua-zt5'])
    request=SnapshotRequest(root,pin);jobs={};main_wave=Path(a.main_wave)
    while True:
        failed=[]
        for name in jobs:
            result=root/'jobs'/name/'result.json'
            if result.exists() and read(result).get('exit_code')!=0:failed.append(name)
        if failed:
            if any(not (root/'jobs'/name/'result.json').exists() for name in jobs):
                time.sleep(30)
                continue
            write(root/'FAILURE.json',dict(status='FAILED',jobs=failed,automatic_retry=False))
            return
        if len(jobs)==reg['world'] and all((root/'jobs'/n/'result.json').exists() for n in jobs):break
        reserved,count=occupied_main(main_wave)
        # Avoid races with the main owner's pre-CUDA reservations. All32 native
        # evaluator jobs must be dispatched (or the wave finished) first.
        if count>=32 or (main_wave/'COMPLETE.json').exists():
            routes,inventory=request.resources.available(exclude=reserved,min_free_MiB=32768)
            write(root/'inventory_latest.json',inventory)
            for rank in range(reg['world']):
                name='rank'+str(rank)
                if name in jobs:continue
                route=request.route(routes,inventory)
                if route is None:break
                directory=root/'jobs'/name;directory.mkdir(parents=True,exist_ok=False)
                module='tools.qwen_world_postcot.diagnose_post_head'
                spec=dict(source=str(source),source_sha=head,host=route['host'],gpus=[route['gpu']],
                    gpu_uuids=[route['uuid']],min_free_MiB=32768,world_shared_lease=True,module=module,
                    command=[request.resources.MODEL_PYTHON,'-u','-m',module,'--root',str(root),'--rank',str(rank)],
                    receipt_dir=str(directory))
                write(directory/'spec.json',spec);request.jobs[name]=spec;jobs[name]=spec
                write(directory/'operator.json',request.resources.launch(directory/'spec.json',str(source)))
                # Reservation belongs to the actual source request and exact
                # worker receipt; route() also masks this request's active GPUs.
                routes=[r for r in routes if r['uuid']!=route['uuid']]
        write(root/'status.json',dict(at=time.time(),stage='RUNNING' if jobs else 'WAITING_MAIN_ADMISSION',
            launched=list(jobs),main_launched=count,main_reserved_physical_gpus=len(reserved)))
        time.sleep(30)
    from tools.qwen_world_postcot.diagnose_post_head import summarize
    result=summarize(root);report=Path(a.report_dir);report.mkdir(parents=True,exist_ok=True)
    write(report/'analysis.json',result)
    write(report/'registration_summary.json',{k:v for k,v in reg.items() if k not in ['tokens','natural_tokens','targeted_tokens']})
    write(report/'training_loss_windows.json',read(root/'training_loss_windows.json'))
    write(report/'qualification.json',read(root/'qualification.json'))
    lines=['# 动作后PDMS分项预测：IL60k独立诊断','',
        '这是训练辅助head的准确性，不是部署PDMS，也不是修复器/选轨器。固定两条已评分轨迹来自IL40k与IL53k；'
        '预测器来自IL60k，全部参数冻结。当前观测与请求轨迹为输入，评分仅在输出后用于误差统计。', '',
        '自然样本2048场景独立hash抽取；额外NC/DAC/TTC边界差异场景只作定向诊断，分表报告。'
        '没有新增policy rollout、scorer调用或optimizer update；Test已被历史观察。', '',
        '| 分项，固定IL53k轨迹自然样本 | MAE原始0–1 | 平均预测 | 平均真值 | 违规recall@0.5，仅硬0/1 | 违规AUROC，仅硬0/1 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in result['groups']['natural']['by_candidate']['1'].items():
        b=row.get('hard_binary',{});auc=b.get('AUROC');recall=b.get('violation_recall_at_half')
        lines.append(f"| {name} | {row['MAE']:.6f} | {row['mean_prediction']:.6f} | {row['mean_target']:.6f} | {recall if recall is not None else 'N/A'} | {auc if auc is not None else 'N/A'} |")
    lines += ['', 'NC=0.5等软标签保留回归/BCE统计，硬二分类单独剔除并列出数量；硬子集无双类时AUROC为null。'
              '简单accuracy必须与always-pass基线和违规recall一起读。EP按连续目标解释。', '',
              'factor使用soft-label BCE，完美预测仍有目标熵下界；analysis同时列出BCE下界与excess BCE。'
              'post总训练loss=factor+0.25×NC/DAC pair，再在总IL loss中乘0.2。', '',
              '完整两轨迹、自然/定向分层、pair排序和组合PDMS误差、有效分母及成本见analysis.json。'
              '组合预测是分项估计值的乘积，不能视为精确的期望联合PDMS；不能由预测准确自动推出其辅助梯度提高了实际策略。', '',
              '模型、数据、只读head与评分身份和真实入口见registration_summary.json；原逐场景输出仅留本地。', '']
    (report/'README.md').write_text('\n'.join(lines))
    def git(*args):return subprocess.check_output(['git',*args],cwd=source,text=True)
    if git('rev-parse','HEAD').strip()!=head:raise ValueError('Changed source during diagnostic')
    prefix=str(report.relative_to(source))
    for row in git('status','--porcelain=v1','-z','--untracked-files=all').split('\0'):
        if row and not row[3:].startswith(prefix+'/'):raise ValueError('Unrelated publication changes')
    subprocess.run(['git','add','--',prefix],cwd=source,check=True)
    subprocess.run(['git','commit','-m','Record held-out World post-head prediction audit'],cwd=source,check=True)
    branch=git('branch','--show-current').strip();commit=git('rev-parse','HEAD').strip()
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=source,check=True)
    if git('ls-remote','origin','refs/heads/'+branch).split()[0]!=commit:raise ValueError('Remote mismatch')
    write(root/'PUBLICATION.json',dict(status='PUBLISHED',commit=commit,remote_SHA=commit))
    write(root/'COMPLETE.json',dict(status='COMPLETE',scenes=result['logical_scene_count'],all_owned_workers_exited=True))


if __name__=='__main__':main()
