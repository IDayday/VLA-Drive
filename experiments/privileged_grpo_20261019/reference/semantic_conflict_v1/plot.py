import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;a=json.loads((ROOT/'ANALYSIS.json').read_text())
fig,ax=plt.subplots(figsize=(8,2.7),layout='constrained')
for i,n in enumerate(['F-C','F-D','F-A']):
    v=a['contrasts'][n]['pdms_mean'];m=v['mean'];l,h=v['bootstrap95'];ax.errorbar(m,2-i,xerr=[[m-l],[h-m]],fmt='o',color='#226b8e',capsize=5)
ax.axvline(0,color='#888',lw=1);ax.set_yticks([2,1,0],['F - C (primary)','F - D','F - A']);ax.set_xlabel('Paired PDMS difference (scene bootstrap 95% CI)');ax.grid(axis='x',alpha=.18);ax.set_ylim(-.6,2.6);ax.set_title('Semantic-conflict trajectories: 32 scenes, 3 paired seeds, 8 candidates')
fig.savefig(ROOT/'effects.svg');fig.savefig(ROOT/'effects.png',dpi=160)
