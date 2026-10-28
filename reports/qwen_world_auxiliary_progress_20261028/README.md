# Qwen-World auxiliary learning quality

An actual frozen-model audit completed20k/40k/60k on the same64 TRAIN and64 DEV scenes. It performed384 observation forwards, zero policy rollouts, zero scorer calls and zero optimizer updates in238.278seconds; peak allocated10.16GiB. DEV here diagnoses auxiliary representations, not planning performance or method selection. The fixed-cohort readout is distinct from changing-batch training losses.

| Readout, DEV |20k|40k|60k|
|---|---:|---:|---:|
|Current DINO feature MSE|0.019731|0.017146|0.016030|
|Current DINO feature cosine|0.832725|0.855849|0.865690|
|Future feature MSE|0.104900|0.098165|0.095262|
|Future feature cosine|0.721697|0.742291|0.751268|
|Matched visible-object class accuracy|88.260%|88.470%|88.679%|
|Matched object XY mean distance|17.961m|15.829m|14.869m|
|Visible-positive recall within2m, right class and existence≥0.5|2.166%|2.725%|2.935%|

Current and future feature reconstruction improve on both fixed cohorts and beat a position-specific constant baseline fitted only on TRAIN64. DEV20k→60k DINO MSE decreases18.76%, future9.19%. This supports representation learning, not a causal PDMS gain.

Object geometry remains poor despite falling object loss and high matched class accuracy. The small improvement in visible-positive recall does not constitute a useful complete detector. Hungarian assignment includes geometry and class terms; matched class accuracy is conditional on assignment, not independent classification accuracy. Unmatched predictions lack complete negative labels (coverage_valid=False), so precision and complete detection accuracy are undefined. No artificial negative labels were inserted.

The separate60k post-PDMS head audit remains a warning: at threshold0.5 it detected0/37 hard NC failures and0/132 TTC failures on the fixed natural IL53k candidate sample. Falling post loss is not reliable rare-failure detection. That audit used already scored candidates and is documented in reports/qwen_world_post_head_accuracy_20261028 on its published analysis branch; it is not a new planning score.

No auxiliary-removal ablation has established which auxiliary task improves final planning. Full Navtest single-candidate PDMS remains the planning criterion. No active training source or losses were changed by this audit.

```bash
python -m tools.qwen_world_postcot.audit_auxiliary_progress --output /mnt/project/qwen-world-postcot-20261025/aux_progress_20261028/measurements
python -m tools.qwen_world_postcot.summarize_auxiliary_progress --input /mnt/project/qwen-world-postcot-20261025/aux_progress_20261028/measurements --output reports/qwen_world_auxiliary_progress_20261028
```

The first command was run once through the owned GPU worker. Its output directory must not be reused for a second inference. The summary command reads completed artifacts and performs no inference. summary.json/CSV contain TRAIN and DEV aggregates, source hashes, units, accounting and limitations; raw scene records stay private.
