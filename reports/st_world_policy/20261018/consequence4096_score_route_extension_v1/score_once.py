"""Run the unchanged fixed-dev CPU scorer on the authorized zt4 host.

Only the dispatch host allowlist is extended. Fresh live policy, authorization,
route expiry, environment, export completeness, locks, invocation and scoring
budgets remain those of the exact reviewed v5 launcher. No arbitrary route,
new execution lease, automatic retry, GPU context, or scoring definition.
"""
import ast
import hashlib
from pathlib import Path

SOURCE = Path('/mnt/project/st-world-policy-20261016/resources/launch_completed_dev_score_v5.py')
SOURCE_SHA = '2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3'
OLD_HOSTS = ('training-rl-zt2', 'training-vla-zt', 'training-vlawm-tsinghua-zt')
ADDITIONAL_HOST = 'training-vlawm-tsinghua-zt4'


def prepare(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('The original reviewed score launcher changed')
    tree = ast.parse(raw, filename=str(SOURCE))
    statements = [n for n in tree.body if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'ROUTE_HOSTS_V3']
    if len(statements) != 1 or ast.literal_eval(statements[0].value) != OLD_HOSTS:
        raise ValueError('Original route allowlist is not the reviewed one')
    statements[0].value = ast.parse(repr(OLD_HOSTS + (ADDITIONAL_HOST,)), mode='eval').body
    return ast.fix_missing_locations(tree)


if __name__ == '__main__':
    # The launch receipt records this actual adapter hash through __file__.
    namespace = dict(__name__='__main__', __file__=str(Path(__file__).resolve()))
    exec(compile(prepare(SOURCE.read_bytes()), str(SOURCE), 'exec'), namespace)
