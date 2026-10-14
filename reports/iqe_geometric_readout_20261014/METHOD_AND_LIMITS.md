# Why this single follow-up

The completed score-objective/dropout matrix did not establish reliable heldout
selection. Its best dev gain was +0.1658 percentage points with a paired-log
interval crossing zero. The prior global and trajectory-conditioned global
visual readers changed no choices when V was mismatched on the fixed64-scene
fitting diagnostic. These observations motivate a more explicit association
between a candidate and current image regions; they do not prove that spatial
reading is the sole cause of poor generalization.

[SparseDriveV2](https://arxiv.org/html/2603.29163v1), sections3.4–3.5 and table6,
uses path/trajectory-conditioned feature aggregation and reports an advantage
for deformable aggregation over ordinary cross-attention in its setting.
Its metric labels come from a rule-based evaluation system. This supports
testing a geometrically aligned readout, but its architecture, candidate
vocabulary, backbone and v2 ablation metric differ from ours. We import no
weights, teacher predictions, vocabulary or benchmark results into IQE.

Our earlier rich-bank audit already found1,457 close safety-crossing pairs
at XY ADE<=0.1m, across1,141 D_S scenes and139 logs. More negative rows alone
are therefore not a demonstrated cure. The current full trial keeps that exact
bank and the original S11 objective, scenes, initialization and update budget.
Only current camera calibration and candidate-local visual readout change.
Calibration is used in the geometric lookup, not added as an expert/log ID.

The small real-data result supports proceeding to one full trial: 96.4976
training PDMS versus94.1814 for the matched global readout; wrong V reduces
it to88.5503 and changes37/64 choices. This is not heldout evidence. The full
run is independently initialized and cannot load these diagnostic weights.

Known limitations remain: fixed z=0/1m anchor hypotheses are not measured
ground height; pinhole projection matches the locked official convention but
does not prove lens rectification; projected points do not imply visibility,
occlusion knowledge or access to future traffic. The reader has no additional
encoder. All original W/A/ego context remains available when projected points
are outside the camera views. Candidate attention remains independent.

The actual single-VLA image path passed four real-scene comparisons. A separate
two-rank, fresh-process diagnostic then crossed the natural/ranking switch and
matched continuous training exactly in parameters, optimizer, scheduler, both
sampler streams, RNG, next IDs, losses and gradient norms. It used eight
disposable updates, not extra production model selection.

Current scientific next decision is based on the registered complete trial:
stage_val selection at3000/4000/5000/6000, optional selector_cal margin, then
the complete fixed development population. Raw argmax, safety regressions,
best fixed-policy comparison and wrong-V diagnostics remain visible. Past dev
results have been observed, so this is an exploratory follow-up rather than a
new untouched-test claim. No new navtest or ACTIVE update is scheduled.
