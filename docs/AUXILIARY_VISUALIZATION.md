The auxiliary visualization entry point is `python -m tools.action_video_foresight.visualize_auxiliary`.
It performs no optimizer updates and uses complete 100k checkpoints restored from FP32 masters (TF32 disabled).

1. `prepare --output OUT --plan FORMAL_PLAN --teacher-data GT_RECORDS --c1-run C1_RUN --legacy-representations C1_REPRESENTATIONS --gallery-size 10`
2. `infer --output OUT --arm C1` and separately `--arm S0`, `S1`, `S2`, `S3`, `S4` on available GPUs.
3. `render --output OUT`; open `OUT/gallery/index.html` locally.

Publish numeric evidence with `python -m tools.action_video_foresight.summarize_visualization --artifacts OUT --output REPORT_DIR`. After reviewing the actual images, a case-note file can create a Chinese local overview with `python -m tools.action_video_foresight.annotate_visualization --artifacts OUT --notes CASE_NOTES_JSON`; open `OUT/gallery/analysis.html`. The latter only adds explanatory pages and current RGB thumbnails; no model forward, optimizer update or change to scores.

Selection balances navigation, current ego speed, selected GT peer motion and clip availability; log preference and hashes resolve ties. It never uses prediction errors. The default `infer --scope representatives` computes only the selected cases; it cannot support full-development statistical conclusions. Explicit `--scope full_dev` evaluates all development scenes. A smaller attribute-selected gallery can reuse the same train-fit display basis with `prepare --reuse-preparation PARENT --representative-indices 0,2,5,7,9,10,17,19,21,22 --gallery-size 10 --output OUT`. This selection uses only scene properties and source RGB, before any model outputs. GT vehicle centres are offline diagnostic anchors only, not student input or dense semantic labels.

Current alignment uses the original raw post-norm DINO feature scale. S-series future comparisons use their actual channel-LayerNorm loss space. Their training-only arithmetic mean templates are not normalized a second time. C1 future comparisons retain the legacy raw-feature scale. Video static references are genuine same-encoder current-repetition clips; they are artificial diagnostic controls, never training targets. DINO static references use the same input384x288/pool2 recipe as the future frames.

The three PCA colours are display-only, fitted on 64 deterministic training scenes and fixed across teacher, student, means and static controls. They are not reconstructed RGB, a target compressor, or a new model component. Vehicle affinity maps use the same teacher anchor for all displayed outputs. The current grid is genuinely6x8; nearest interpolation exposes its coarse resolution.

Source sensor pictures, individual scene identities, features and HTML galleries remain outside git. Only code, aggregate statistics, anonymized numeric illustrations and written conclusions may be published. The gallery preserves missing clips and displays native time intervals. Video tokens have whole-clip context and are not independent causal frames. These diagnostics are auxiliary capabilities, not new PDMS scores or evidence of causal interaction.
