"""Validate every declared shard and merge all rows without filtering failures."""
import argparse,csv,json
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--shards',type=int,required=True);a=p.parse_args();root=Path(a.root)
 expected=json.loads(Path(a.manifest).read_text());rows=[];manifests=[]
 for shard in range(a.shards):
  m=json.loads((root/f'manifest_{shard}.json').read_text())
  if m['arguments']['shard']!=shard or m['arguments']['shards']!=a.shards:raise ValueError('Shard contract mismatch')
  r=list(csv.DictReader(open(root/f'scenes_{shard}.csv')))
  if len(r)!=m['samples'] or [x['token'] for x in r]!=expected[shard::a.shards]:raise ValueError('Shard scene coverage/order mismatch')
  rows+=r;manifests.append(m)
 if len(rows)!=len(expected) or len({r['token'] for r in rows})!=len(expected):raise ValueError('Missing or duplicate scenes')
 order={t:i for i,t in enumerate(expected)};rows.sort(key=lambda r:order[r['token']]);keys=sorted({k for r in rows for k in r})
 with (root/'scenes.csv').open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 result={'samples':len(rows),'failed':sum(r['status']!='ok' for r in rows),'shards':a.shards,'manifests':manifests};(root/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='manifests'}))
if __name__=='__main__':main()
