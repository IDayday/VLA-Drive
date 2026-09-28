# 64-scene camera learning diagnostic: all arms complete512updates

These are TRAINING-set fit results on the same64scenes/60logs. Each arm has512verified effective updates,8192scene presentations, globalbatch16, and one fixed camera-only inference seed42. Every arm retains64/64scene rows with0failures. Formal100000-step models initialize afresh and are still running. No full-development/Navtest PDMS result exists.

| Arm | Ego ADE@64 | Ego ADE@256 | Ego ADE@512 | Ego FDE@512 | Current matched vehicles@512 | Joint-selected matched vehicles |
|---|---:|---:|---:|---:|---:|---:|
|A Base|7.317|1.438|0.836|1.129|not a vehicle detector|not applicable|
|B Joint|10.037|6.505|3.448|3.527|14/333|12/333|
|C Joint-Mask|10.707|6.748|3.800|3.960|11/333|9/333|

Distances are meters; lower is better. The333vehicle targets belong to the supervised current ROI/FOV, from1381source vehicles. The2mcurrent matching gate is an evaluation rule, never a training-supervision gate. Low coverage is a central limitation; errors on a handful of matched vehicles cannot stand for all vehicles.

Learning is real: first16versuslast16mean FM loss decreases A1.684→0.079, B1.676→0.433, C1.676→0.439. B/C box loss falls approximately0.180→0.043 and auxiliary motion loss0.937→0.394/0.397. Main FM contains extra vehicle terms in B/C, so its absolute magnitude is not a like-for-like A/B comparison. Ego error still improves from256to512; sufficient convergence is not established.

On the SAME64targets, C−B ego ADE is+0.352m, log-cluster95%interval[+0.218,+0.489]; FDE is+0.434m[+0.212,+0.658]. This is unfavorable to MASK on this fixed training diagnostic, not a cross-seed or held-out conclusion. B−A ADE is+2.612m[+1.884,+3.370].

For the8vehicles jointly matched by both models, B/C joint ADE is13.476/12.050m and FDE17.446/14.731m. C improves FDE on that tiny intersection, while its overall coverage is lower. ADE-difference interval crosses zero. This selective intersection cannot establish general vehicle-prediction or planning benefit. Full per-seed denominators and stationary/moving groups remain in metrics.json.

C actually used1074role-completion scene presentations:528ego-hidden and546neighbor-hidden;718of1792auxiliary presentations fell back to all-hidden. Main-task exposure remains8192for each arm. B has1792matched extra all-hidden presentations and zero applied role tasks. Fixed final64updates disable auxiliaries for both.

Current evidence: basic learning is present; joint models fit ego worse than Base; vehicle localization/coverage and joint future prediction remain weak; no overall MASK advantage is established. The extra-task/interference explanation is a hypothesis, not a diagnosed cause. Full training continues with the locked recipe, without selecting favorable scenes or changing Navtest protocol.

Earlier575optimizer calls in superseded diagnostics were confirmed no-ops and are not included as learning. Their raw ledgers and costs remain preserved under optimizer_stasis. All1536updates in the corrected three-arm diagnostic passed per-step FP32-master update probes. The old learning_curves figure used inconsistent automatic colors on panels where A is absent; this report regenerates stable arm colors. Underlying values and training code are unchanged.

Evidence: metrics.json, paired_analysis.json, paired ego CSVs, and learning_curves.png. One training seed and one inference seed are insufficient for a final algorithm conclusion.
