"""Build and audit the full 103,288 NAVTRAIN current-observation population."""
import argparse
import json
from starVLA.rl.qwen_world_research_data.manifest import build_population

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--reuse-asset-audit',help='Sealed exact-population prior existence audit; recheck only its missing cache scenes')
    a=p.parse_args()
    audit=build_population(a.output,verify_files=True,reuse_asset_audit=a.reuse_asset_audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('missing_current_images','missing_metric_caches')},indent=2))

if __name__=='__main__':main()
