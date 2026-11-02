# Integrated SafetyEP: actual 10k commitment and 2k/5k/10k evaluation

User requires 10,000 actual optimizer updates before judging a longer continuation. Future full NAVTEST uses only actual updates 2,000, 5,000 and 10,000. The completed 250-update result remains historical. The existing 625-update native stage is retained only as a checkpoint handoff.

CPU controller source: frozen `runtime/budget_v4` at `f975bd3`. Scientific source `2b9f320` and GPU execution source `d745fcd` are unchanged. Same IL70 lineage, Adam/RNG/history bank and fixed 12,500-update warm125/keep625/cos2500/floor0.1 LR clock continue. No GPU process was signalled or reset; all eight rank journals advanced across adoption.

The committed budget is 160,000 scene exposures and 2,560,000 fresh PG candidates, plus separately accounted TRAIN B1 verification queries. This is not equal compute to the original Control.

Two focused CPU tests passed, plus the real frozen registry/factory validation. See attached receipts for applied configuration and actual process continuity.
