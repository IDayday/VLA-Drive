"""Create a per-host overlay without modifying existing training environments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-python', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--packages', type=Path, required=True)
    args = parser.parse_args()
    mmcv = args.packages/'mmcv-2.2.0-cp310-cp310-linux_x86_64.whl'
    expected = '7b2daaa4170c145318adad6105d0a87ab1dba5dd53046c54bc56fc1bdcbe1d02'
    if hashlib.sha256(mmcv.read_bytes()).hexdigest() != expected:
        raise ValueError('Wrong pinned MMCV CUDA wheel')
    nusc = args.packages/'nuscenes_devkit-1.2.0-py3-none-any.whl'
    provenance = json.loads((args.packages/'NUSCENES_BUILD.json').read_text())
    if hashlib.sha256(nusc.read_bytes()).hexdigest() != provenance['wheel_sha256']:
        raise ValueError('Wrong fixed-source nuScenes wheel')
    python = args.environment/'bin/python'
    if not python.exists():
        subprocess.run([str(args.base_python), '-m', 'venv', '--system-site-packages', str(args.environment)], check=True)
    required = [str(mmcv), str(nusc), 'mmengine==0.10.7', 'addict==2.4.0',
                'yapf==0.43.0', 'termcolor==3.3.0', 'descartes==1.1.0']
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', *required], check=True)
    check = 'import torch,transformers,deepspeed,mmcv,nuscenes; assert torch.__version__=="2.5.1+cu124"; assert transformers.__version__=="4.57.0"; assert deepspeed.__version__=="0.16.9"; assert mmcv.__version__=="2.2.0"'
    subprocess.run([str(python), '-c', check], check=True)
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (args.environment/'installed_versions.txt').write_text(freeze)
    identity = {'kind': 'isolated_system_site_overlay', 'base_python': str(args.base_python),
        'environment': str(args.environment), 'MMCV_wheel_sha256': expected,
        'nuscenes': provenance, 'packages': required, 'shared_training_environment_modified': False}
    (args.environment/'ENVIRONMENT_IDENTITY.json').write_text(json.dumps(identity, indent=2)+'\n')


if __name__ == '__main__':
    main()
