"""Run the unchanged fixed-dev exporter on the additional authorized zt4 host.

Only the dispatch host allowlist is extended. Fresh live policy, authorization,
environment, GPU UUID/idle checks, locks, budgets, all artifact checks and native
inference remain those of the exact reviewed v5 launcher. No arbitrary route.
"""
import ast
import hashlib
from pathlib import Path

SOURCE = Path('/mnt/project/st-world-policy-20261016/resources/launch_formal_dev_export_v5.py')
SOURCE_SHA = 'ad0ee24987bdba9ee3fbc46a5efbdb1445b0a6a69d4d588fc32e960738b876bf'
OLD_HOSTS = ('training-rl-zt2', 'training-vla-zt', 'training-vlawm-tsinghua-zt')
ADDITIONAL_HOST = 'training-vlawm-tsinghua-zt4'


def prepare(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('The original reviewed export launcher changed')
    tree = ast.parse(raw, filename=str(SOURCE))
    statements = [n for n in tree.body if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'ROUTE_HOSTS_V3']
    if len(statements) != 1 or ast.literal_eval(statements[0].value) != OLD_HOSTS:
        raise ValueError('Original route allowlist is not the reviewed one')
    statements[0].value = ast.parse(repr(OLD_HOSTS + (ADDITIONAL_HOST,)), mode='eval').body
    return ast.fix_missing_locations(tree)


if __name__ == '__main__':
    # __file__ points here so the launch receipt records this actual adapter hash.
    namespace = dict(__name__='__main__', __file__=str(Path(__file__).resolve()))
    exec(compile(prepare(SOURCE.read_bytes()), str(SOURCE), 'exec'), namespace)
