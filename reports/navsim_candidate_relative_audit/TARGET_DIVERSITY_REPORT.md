# Candidate-relative Target Diversity

- Scenes / candidate pairs: 500 / 33000
- Non-zero pairwise consequence distance: 100.000%
- Mean unique consequences per scene: 12.00
- Saved hard-negative pairs: 5692
- Trajectory vs consequence Spearman: 0.5763255809342668
- Consequence distance vs PDM-score difference Spearman: 0.31799120649538043

The actor component matches stable track hashes and explicitly penalizes actors present in only one nearest-N set, so masks and unequal actor sets do not silently compare unrelated slots.  O(K²) supervision is non-degenerate when the non-zero ratio and per-scene unique counts exceed their trivial values; this report records both rather than inferring diversity from trajectory perturbations alone.

Hard-negative categories include close geometry with collision/TTC or DAC differences, close endpoints with different intermediate consequences, same/close GT prefixes with divergent tails, and geometrically distant candidates with similar evaluation.
