#!/usr/bin/env python3

import os
from pathlib import Path

from navsim.common.dataloader import MetricCacheLoader


def count_feature_tokens(root: Path) -> int:
    return sum(
        1
        for path in root.glob("*/*")
        if (path / "internvl_feature.gz").is_file()
        and (path / "trajectory_target.gz").is_file()
    )


metric_root = Path(os.environ["DRIVEVLA_NAVTRAIN_METRIC_CACHE"])
feature_root = Path(os.environ["DRIVEVLA_NAVTRAIN_FEATURE_CACHE"])
metric_count = len(MetricCacheLoader(metric_root))
feature_count = count_feature_tokens(feature_root)

expected = 103_288
print(f"metric cache:  {metric_count:,} / {expected:,}")
print(f"feature cache: {feature_count:,} / {expected:,}")
if metric_count != expected or feature_count != expected:
    raise SystemExit("Stage-2 cache is incomplete")

