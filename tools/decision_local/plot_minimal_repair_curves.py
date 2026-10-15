"""Plot the fixed common milestones; never select or smooth a best checkpoint."""
import argparse,csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=list(csv.DictReader(a.csv.open()))
    colors=dict(LOG_NONE='#386cb0',LOG_CV='#008b8b',LOG_FUT='#7b4ea3',MIN_NONE='#4a8b45',MIN_CV='#c67b16',MIN_FUT='#bc3434')
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,layout='constrained')
    for row,seed in enumerate((42,43)):
        for col,dist in enumerate(('NATURAL','RECOVERY')):
            ax=axes[row,col];subset=[x for x in rows if int(x['seed'])==seed and x['distribution']==dist]
            if not subset:raise ValueError('Missing common-seed/distribution curve')
            baseline={float(x['input_PDMS']) for x in subset}
            if len(baseline)!=1:raise ValueError('Mixed input baselines')
            ax.axhline(next(iter(baseline)),color='#555555',linestyle='--',linewidth=1,label='Input plan')
            for kind,color in colors.items():
                values=sorted((int(x['epoch']),float(x['PDMS'])) for x in subset if x['kind']==kind)
                if not values:continue
                if [x[0] for x in values]!=[6,12,18,24]:raise ValueError('Incomplete fixed observations')
                ax.plot(*zip(*values),marker='o',markersize=3.5,color=color,linewidth=1.5,label=kind)
            ax.set_title(('Natural S0 plans' if col==0 else 'Fixed recovery plans')+f' | head seed {seed}')
            ax.set_ylabel('PDMS (%)');ax.set_xticks([6,12,18,24]);ax.grid(alpha=.2)
            if row==1:ax.set_xlabel('B-fold training epoch')
    axes[0,1].legend(fontsize=8,ncol=2,loc='center right')
    fig.suptitle('Frozen S0; common 24-epoch endpoints\nCV/FUT are privileged probes; the two input distributions are separate',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('.png','.pdf'):fig.savefig(a.output.with_suffix(extension),dpi=180)
    a.output.with_suffix('.json').write_text(json.dumps({'input_sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),
        'observations':len(rows),'no_smoothing_or_checkpoint_selection':True},indent=2)+'\n')


if __name__=='__main__':main()
