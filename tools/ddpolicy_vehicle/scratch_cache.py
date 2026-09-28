"""Ownership-checked lifecycle for rebuildable evaluation checkpoint copies."""
import argparse
import json
from pathlib import Path
import shutil
from .prepare_data import atomic_json


def manage_cache(cache, predictions, release=False):
    root=Path(cache)
    if root.is_symlink():raise ValueError('Scratch directory cannot be a symlink')
    owner={'kind':'ddpolicy_ephemeral_checkpoint_copy_v1',
           'cache':str(root.absolute()),'predictions':str(Path(predictions).absolute())}
    marker=root/'SCRATCH_OWNER.json'
    if release:
        if not root.exists():return
        if not marker.exists() or json.loads(marker.read_text())!=owner:
            raise ValueError('Refuse to remove a cache without exact campaign ownership')
        bank=Path(predictions);shard=json.loads((bank/'shard_0.json').read_text())
        if shard['status']!='complete' or shard['failed'] or shard['completed']!=shard['requested']:
            raise ValueError('Retain scratch until prediction export is complete')
        # This directory contains only exact, reconstructible checkpoint copies.
        # Original training checkpoints and previous task caches are elsewhere.
        shutil.rmtree(root)
    elif root.exists():
        if not marker.exists() or json.loads(marker.read_text())!=owner:
            raise ValueError('Existing cache is not owned by this prediction job')
    else:
        root.mkdir(parents=True)
        atomic_json(marker,owner)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--release',action='store_true');a=p.parse_args();manage_cache(a.cache,a.predictions,a.release)


if __name__=='__main__':main()
