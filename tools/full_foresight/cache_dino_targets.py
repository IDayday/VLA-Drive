"""Extract all four physical times with the same frozen DINO/pool recipe.

The common sharded writer encodes once per input resolution and emits both
pooling variants. Full-method schema rejects current-only/FLUX cache aliases.
"""
from tools.dino_tradeoff.cache_targets import main

if __name__ == '__main__':
    main(full_method=True)
