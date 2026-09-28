"""Register an already authorized local generic VAE; never downloads or bypasses access controls."""
import argparse
from pathlib import Path
from .flux_targets import FLUX_REPOSITORY,FLUX_REVISION,FLUX_WEIGHT,FLUX_CONFIG,verify_flux_source
from starVLA.model.modules.vehicle_joint.initialization import file_sha256
from .score_pdms import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output)
    if out.exists():raise FileExistsError('Do not overwrite a registered VAE identity')
    identity={'repository':FLUX_REPOSITORY,'revision':FLUX_REVISION,
              'files':{name:file_sha256(root/name) for name in (FLUX_CONFIG,FLUX_WEIGHT)},
              'source':'generic public FLUX.1-schnell image VAE; no driving adaptation',
              'generic_pretraining_data_fully_auditable':False}
    verify_flux_source(root,identity);atomic_json(out,identity)


if __name__=='__main__':main()
