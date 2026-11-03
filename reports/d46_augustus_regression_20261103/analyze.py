"""Read complete, sealed original results and finite journals; never run a GPU."""
import hashlib,json,pathlib,statistics,time
ROOT=pathlib.Path('/mnt/project/d46-priority-rl-20261103')
OUT=ROOT/'analysis/AUGUSTUS_REGRESSION_20261103'
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def sealed(p):
 v=read(p);body=dict(v);sha=body.pop('sha256');assert digest(body)==sha;return v
contract=read(ROOT/'PARENT_CONTRACT.json');base=read(contract['complete_navtest']['summary'])
points=[]
for arm,updates in [('AUG1_FULL',[100,200]),('AUG2_LORA',[200,400])]:
 rows=[json.loads(line)for line in (ROOT/'runs'/f'{arm}_v1'/'metrics_rank0.jsonl').read_text().splitlines()]
 for u in updates:
  d=ROOT/'evaluations'/f'{arm}_u{u:06}';s=sealed(d/'SUMMARY.json');a=sealed(d/'PAIRED_ANALYSIS.json')
  assert s['scenes']==12146 and s['physical_logs']==44 and s['invalid_scenes']==0
  q=next(r for r in rows if r['updates']==u)
  points.append(dict(arm=arm,updates=u,exposure=q['scene_epochs'],PDMS=s['PDMS'],delta_IL46=s['PDMS']-base['PDMS'],
   means=s['means'],paired=a['paired'],gain_loss=a['gain_loss'],summary=str(d/'SUMMARY.json'),summary_file_sha256=file_sha(d/'SUMMARY.json')))
# The original first-gate IO exceptions occurred after successful native seals.
failures=[]
for arm in ['AUG1_FULL','AUG2_LORA']:
 cp=pathlib.Path(read(ROOT/'runs'/f'{arm}_v1'/'latest.json')['checkpoint'])
 seal=read(cp/'COMPLETE');tx=read(cp/'transaction.json')
 assert seal['phase']=='READY'and seal['next_inner']==0
 log=ROOT/'jobs/AUGUSTUS_20261103'/arm/'formal/node0/stdout.log'
 text=log.read_text();assert 'FileNotFoundError'in text and 'evaluation_requests/epoch0.5.json.writing-'in text
 failures.append(dict(arm=arm,checkpoint=str(cp),updates=seal['updates'],native_phase=seal['phase'],failure='missing evaluation_requests parent directory after native0.5E save',source_log=str(log)))
curves={}
for arm in ['AUG1_FULL','AUG2_LORA']:
 rows=[json.loads(l)for l in (ROOT/'runs'/f'{arm}_v1'/'metrics_rank0.jsonl').read_text().splitlines()]
 curves[arm]=[dict(updates=r['updates'],epoch=r['scene_epochs'],KL=r['losses']['reference_KL'],lr=r['lr'][0],
   grad=r['gradient_norm'],fresh_ratio_error=r['losses']['fresh_ratio_error'],clip=r['losses']['clip_fraction'],
   sampled=r['exploration']['random_candidate_mean'],oracle=r['exploration']['oracle_mean'])for r in rows]
 assert all(r['fresh_ratio_error']==0 and r['clip']==0 for r in curves[arm])
value=dict(schema='d46_augustus_regression_evidence_v1',status='DIAGNOSED_CORRECTION_QUALIFYING',at=time.time(),
 baseline_PDMS=base['PDMS'],baseline_means=base['means'],baseline_summary=contract['complete_navtest']['summary'],
 complete_points=points,engineering_failures=failures,curves=curves,confirmed=['First evaluation directory not created',
 'K3 linear training differs from native K10 shift5','M1 fresh ratios equal1 and clipping inactive at gradient evaluation',
 'FULL reference KL grows on changing TRAIN batches','Positive TTC0 canonical credit can coexist with safe alternative'],
 causal_limits='No isolated cause attribution yet. Registered combined correction requires true GPU/B1 results. No new score claimed.',
 corrected_source='/mnt/project/d46-priority-rl-20261103/runtime/augustus_aligned_science_v2',corrected_source_sha='09514510afa29e7fdbce834caffd93f3825eea24',
 corrected_routes=['AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED'],CPU_tests_passed=16,NAVTEST_SELECTED=True,training_test_labels=False)
value['sha256']=digest(value);(OUT/'DIAGNOSIS.json').write_text(json.dumps(value,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'AUG1_FULL':'#be3543','AUG2_LORA':'#2673b3'}
fig,ax=plt.subplots(figsize=(6.6,3.8))
for arm,c in colors.items():
 p=[r for r in points if r['arm']==arm];ax.plot([r['exposure']for r in p],[r['PDMS']for r in p],'-o',color=c,label=arm)
ax.axhline(base['PDMS'],color='#506a62',ls='--',label='IL46 89.496');ax.set(xlabel='Valid TRAIN scene exposures / 103288',ylabel='Complete NAVTEST B1 PDMS',title='Original AUG variants: deployment regression');ax.legend();fig.tight_layout()
fig.savefig(OUT/'deployment_regression.svg');fig.savefig(OUT/'deployment_regression.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.5,3.8))
for arm,c in colors.items():
 rows=curves[arm];axs[0].plot([r['epoch']for r in rows],[r['KL']for r in rows],color=c,label=arm,alpha=.75)
 p=[r for r in points if r['arm']==arm];axs[1].plot([r['exposure']for r in p],[r['paired']['new_zero']for r in p],'-o',color=c,label=arm+' new zero')
 axs[1].plot([r['exposure']for r in p],[r['paired']['repaired_zero']for r in p],'--o',color=c,label=arm+' repaired')
axs[0].set_yscale('symlog',linthresh=.01);axs[0].set(xlabel='Valid scene exposure epoch',ylabel='Coordinate mean reference KL',title='TRAIN drift (different batches)');axs[0].legend()
axs[1].set(xlabel='Valid scene exposure epoch',ylabel='NAVTEST scenes versus IL46',title='New failures exceed repairs');axs[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'trust_and_zero_regression.svg');fig.savefig(OUT/'trust_and_zero_regression.png',dpi=150)
print(json.dumps(dict(status='COMPLETE',points=len(points),diagnosis=str(OUT/'DIAGNOSIS.json'),sha256=value['sha256'])))
