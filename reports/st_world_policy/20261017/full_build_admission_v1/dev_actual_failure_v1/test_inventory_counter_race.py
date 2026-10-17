"""Reproduce only the actual inventory/counter replacement race with 2-byte JSON."""
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import pytest

SOURCE=Path('/mnt/project/VLA-Drive-st-metadata-build-dev-v1-20261017/tools/st_world_policy/prepare_future_metadata.py')

def test_original_inventory_sees_counter_temp_removed_by_atomic_replace(tmp_path):
    # Compile the exact inventory function, without importing/running the builder.
    node=next(n for n in ast.parse(SOURCE.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='metadata_inventory')
    temporary=tmp_path/'CONTROL_BUILD_COUNTERS.json.example.tmp'
    temporary.write_text('{}')
    @contextmanager
    def metadata_parent(path):
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:yield path,fd
        finally:os.close(fd)
    calls=[0]
    def storage_check():
        calls[0]+=1
        if calls[0]==2:
            # scandir yielded the name; writer commits before entry.stat().
            os.replace(temporary,tmp_path/'CONTROL_BUILD_COUNTERS.json')
    namespace={'os':os,'stat':stat,'metadata_parent':metadata_parent,
               'metadata_path':lambda p,**kw:Path(p),'_storage_check':storage_check}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
    with pytest.raises(FileNotFoundError,match='COUNTERS.json.example.tmp'):
        namespace['metadata_inventory'](tmp_path)
    assert (tmp_path/'CONTROL_BUILD_COUNTERS.json').read_text()=='{}'
