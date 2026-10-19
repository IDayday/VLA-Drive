"""Standalone scientific chart; intervals cluster on original drive scenes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
a=json.loads((ROOT/'ANALYSIS.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
names=['B-A','C-B','C-D','D-A','E-C']
values=[a['contrasts'][n]['pdms_mean']['mean'] for n in names]
intervals=np.asarray([a['contrasts'][n]['pdms_mean']['bootstrap95'] for n in names])
y=np.arange(len(names))
axes[0].axvline(0,color='.55',linewidth=1)
axes[0].errorbar(values,y,xerr=np.stack([np.asarray(values)-intervals[:,0],intervals[:,1]-np.asarray(values)]),fmt='o',capsize=4,color='#245c83')
axes[0].set(yticks=y,yticklabels=names,xlabel='Paired PDMS difference (scene bootstrap 95% CI)',title='Five planned contrasts')
axes[0].invert_yaxis()
for i,arm in enumerate('ABCDE'):
    scores=[s['pdms_mean'] for s in a['per_seed'][arm]]
    axes[1].scatter(np.full(3,i)+[-.1,0,.1],scores,color='#8298ac',s=28)
    axes[1].plot([i-.22,i+.22],[a['aggregate'][arm]['pdms_mean']]*2,color='#245c83',linewidth=3)
axes[1].set(xticks=range(5),xticklabels=list('ABCDE'),ylabel='Mean PDMS on 32 scenes',title='Three paired seed batches per condition')
axes[1].set_ylim(0,1)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.18)
fig.savefig(ROOT/'paired_effects.svg')
fig.savefig(ROOT/'paired_effects.png',dpi=160)
plt.close(fig)
