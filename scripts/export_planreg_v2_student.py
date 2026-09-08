import argparse
import json
from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import export_student

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    a = p.parse_args()
    print(json.dumps(export_student(a.input,a.output),indent=2))
