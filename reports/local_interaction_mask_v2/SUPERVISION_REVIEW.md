# Correction: annotations exist; this pipeline loses their supervision

The user correctly challenged the phrase “no supervision.” NAVSIM contains current3D boxes, track IDs and future log observations. The current experiment's89.433%empty neighbor-mask tasks do **not** mean89.433%of NAVSIM lacks future annotations. They describe whether a selected **predicted slot** receives a valid loss after this implementation's association and filtering.

The read-only audit of all7284training scenes partitions every61,894active predicted neighbor occurrence:

| Mutually exclusive outcome | Occurrences |
|---|---:|
| No full-slot Hungarian assignment | 33191 |
| Assigned, rejected by distance only | 18953 |
| Assigned, rejected by distance and class | 1852 |
| Assigned, rejected by class only | 613 |
| Current association accepted, all future points invalid | 45 |
| Current association accepted, at least one future point valid | 7240 |

Of21418rejected assignments,21213refer to GT tracks that have future labels. Of7285accepted current associations,7240(99.3823%)have future labels. Across the full original-slot assignments,143922/146445have future labels. Counts are scene-actor occurrences, not unique physical actors. `SUPERVISION_FLOW.json` and its complete sceneCSV preserve these denominators; source/run provenance is in `SUPERVISION_FLOW_RUN.json`.

An unassigned prediction is not automatically a dataset annotation failure or a proven false positive. Excess hypotheses, the supervised region/capacity, predicted geometry, and the global matching allocation can contribute. This partition localizes the loss of supervision; it does not independently identify the causal contribution of each component. The existing raw-log rebuild/geometry checks give additional evidence, but do not by themselves prove every association is correct.

## What WCog establishes, and what it does not specify

[WCog-VLA §3.4](https://arxiv.org/html/2607.08375v1#S3.SS4) describes current3D perception pretraining, followed by VLM/world-head training with GT boxes **and future trajectories** (equation6), then freezing the VLM for joint trajectory generation. Thus surrounding-agent prediction is an explicitly supervised task. The paper's main text does not specify a complete query-to-GT matching implementation or establish that it uses this experiment's2m/current-class acceptance rule.

The present public-foundation recipe trains current class/box supervision and egoFM, while its legacy motion head is frozen. It then freezes the shared representation and gives the separate local graph only accepted predicted-instance futures. That is a material difference from training the shared agent representation with box and future losses before freezing it. The sparse-supervision result diagnoses this particular recipe and graph quality; it cannot refute supervised neighbor forecasting or WCog's mechanism.

## Training targets versus deployment inputs

Using GT current boxes to assign output queries in the **loss**, then selecting the same GT track's future labels, is standard supervised learning and is allowed by the input contract. Current class/box training already uses full-slot assignment without the later2macceptance gate. That gate is introduced in `local_targets.py` for the frozen predicted graph's trajectory labels.

Feeding GT boxes into the deployed graph, using GT future validity to choose mask targets, or silently repairing identities with GT would change the deployment problem. Those operations remain excluded. A separate privileged-current teacher diagnosis would need an explicit label and its own results; none has been run here.

A meaningful subsequent training correction must distinguish assignment needed to teach immature queries from a quality criterion used to trust frozen predicted identities. It should establish useful current-plus-motion representations before depending on them for local interaction learning, and separately measure detector quality and association losses. Blindly removing the gate can attach the wrong vehicle's future to a prediction; leaving it unchanged while calling empty losses “missing data” is also unjustified.

The registeredALL/MASK/nearest32pass results are preserved. No threshold, matching rule, graph, seed or active training job was silently changed after these findings. The now-paused P3controls remain a bounded evaluation of that explicitly limited recipe, not evidence that supervision data are unavailable. Any new training recipe must have separate provenance and matched controls, and must be frozen before its own Navtest evaluation.
