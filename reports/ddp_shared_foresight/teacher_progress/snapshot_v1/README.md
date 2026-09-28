# In-progress full teacher snapshot

The randomly initialized full-data teacher remains RUNNING from frozen d3c05d136949ecc14b1a8312fa12bf966ae03643. This snapshot stops at2755updates/705040presentations; it does not stop or replace the live training. Full planned length remains30epochs/11910updates. No checkpoint is frozen early.

Fixed development milestones0/1/2/4 contain1696scenes,16logs and7342queries each, zero failures. At epoch4,1033ego queries with peer futures have ADE0.88779m vs1.03960m with peer futures removed;5646vehicle queries have3.26510m vs5.21379m. The paired full-minus-removed log-cluster intervals are[-.20281,-.10119]m and[-2.29195,-1.48855]m. Removing peer futures can be out of distribution; these are teacher mechanism diagnostics, not causal proof or camera-only planning results. Ego-only663queries show identical predictions in both conditions and are retained.

Nominal target-role proportions are50/50. Actual ego proportion67.06%includes17.06%fallback when a valid neighbor is unavailable; this is explicitly counted, not hidden. Summary groups retain stationary/moving vehicles, complete/partial labels and with/without-peer queries.
