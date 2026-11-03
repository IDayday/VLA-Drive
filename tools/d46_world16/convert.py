"""CPU-only READY conversion to a NEW execution root; old files unchanged."""
import argparse,json
from starVLA.rl.d46_world16.native import prepare

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--qualification',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.checkpoint,a.output,qualification=a.qualification),indent=2))

if __name__=='__main__':main()
