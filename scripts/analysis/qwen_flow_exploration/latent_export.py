"""Export pure small-q deployment weights; never exports optimizer/base/cache."""
import argparse
import json
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.latent.deployment import export_q_package,load_q_package


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--allow-qualification',action='store_true',help='CPU/engineering acceptance only; never formal evaluation')
    args=parser.parse_args()
    protocol=read_protocol(args.protocol)
    metadata=export_q_package(args.checkpoint,protocol,args.output,allow_qualification=args.allow_qualification)
    _,loaded=load_q_package(args.output,protocol,allow_qualification=args.allow_qualification)
    assert loaded==metadata
    print(json.dumps({key:metadata[key] for key in ['arm','training_seed','qualification','policy_sha256','checkpoint_state','base_model_sha256','protocol_sha256']}))


if __name__=='__main__':main()
