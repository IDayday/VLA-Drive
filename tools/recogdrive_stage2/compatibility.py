"""Narrow annotation-only compatibility for official code on Python 3.9.

The official DiT uses a Python 3.10 union return annotation. Postponing that
annotation changes neither executable model code nor the checkout on disk.
"""
import __future__
import importlib.abc
import importlib.util
from pathlib import Path
import sys


def install_annotation_compatibility(source):
    if sys.version_info >= (3, 10):
        return 'not_required'
    name = 'navsim.agents.recogdrive.recogdrive_dit'
    path = Path(source) / 'navsim/agents/recogdrive/recogdrive_dit.py'
    if name in sys.modules:
        raise RuntimeError('Install annotation compatibility before importing official DiT')

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            code = compile(path.read_bytes(), str(path), 'exec',
                           flags=__future__.annotations.compiler_flag,
                           dont_inherit=True)
            exec(code, module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name:
                return importlib.util.spec_from_file_location(fullname, str(path_file), loader=Loader())
            return None

    path_file = path
    sys.meta_path.insert(0, Finder())
    return 'python39_postponed_annotations_only_official_recogdrive_dit'
