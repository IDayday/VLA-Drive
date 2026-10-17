# ST teacher-channel control, 256 versus 384

Design ID: `ST-TEACHER-CHANNEL-256-384-001`. Source: `951711fbd2eae1730262a79575b08024bcd2001d`. Status: design recorded before configuration implementation; independent review and formal launch approval are not supplied by this card. Owned paths are this report directory, `configs/st_world_policy/teacher_channel_control_v1/`, and a narrow configuration behavior test.

The user's clarification concerns the image supplied to the pretrained **teacher**, not the student. This pair holds the teacher at width384/height288 with its native18x24 grid, and holds the student's original three1024x576 current images fixed. It changes only the number of retained teacher PCA channels. This is a new explicit two-arm study; it does not reopen the entire spatial/temporal matrix or alter any running experiment.

## Question and falsifiable hypothesis

With the native432 spatial positions retained, does256-channel supervision retain useful learned spatial information and planning performance compared with384, while reducing projected-target/readout cost? The hypothesis is that the discarded channels are not a material bottleneck at this spatial resolution. It is contradicted by reproducible deterioration in learned-W spatial readouts or matched planning/safety outcomes; raw PCA reconstruction error alone neither proves nor disproves the downstream hypothesis. No noninferiority margin or statistical success is invented in this preparation.

## Evidence and reuse

The existing evidence card `DINO_PCA_NONWHITEN` in `reports/st_world_policy/20261016/evidence/DESIGN_EVIDENCE_CARDS.json` records the previously read DINO-Foresight paper, sections3.1–3.4 and appendixA.1, and official code commit `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0`. Its fixed local `pca.py` and `src/dino_f.py::pca_transform/pca_inverse_transform` were reread: training-only channel standardization, nonwhitened PCA, and the same inverse scales. Original multi-layer DINOv2 is not our single-layer DINOv3. The proposed256/384 widths are project choices, not an official optimum.

Existing implementation already provides the complete channel path:

- `teacher_pca.TeacherPCA.transform`: unchanged mean/std/centering and prefix of one frozen basis.
- `st_world_dataset.projected_teacher_contract`: a different content identity for each output width, retaining raw teacher geometry and source identity.
- `STWorldTrainingDataset`: loads PCA with `dimension=expected_contract.channels`, verifies the effective contract, and never relabels raw1024 data as compressed data.
- `SpaceTimeConfig` and `SpaceTimeStudent`: configure the existing sparse readout output width from that contract. W tokens, sparse positions, action head and original Flow objective remain unchanged.
- `train_world` and `check_formal_approval`: existing registration, strict loading, optimizer, sampling, checkpoints and independent review gates remain the only training path.

The frozen basis identity is `268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46`, fitted from one train-only32768-patch plan. The namespace `/mnt/project/st-world-policy-20261016/audits/history_data/pca_fixed384_full_current_v1` contains **both** dimensions `[256,384]`; its name is historical, not a limitation. `DIMENSION_SELECTION.json` SHA256 `820f4a33e1ccace8fa7bbd1b557a10debfbeba556ebff069033f136e368ac275` records the prior384 choice and closed search at2026-10-16T21:39:42Z. It is preserved, not rewritten; this newly requested pair gets separate run identities.

Teacher-feature probes already observed, on the same64train/16dev population:

| Metric | 256 | 384 |
|---|---:|---:|
| Dev relative reconstruction SSE |0.13670651|0.09361588|
| Dev depth MAE, m |5.332822|5.197710|
| Small-vehicle region PR-AUC |0.916009|0.914696|

Depth has15922 valid patches; vehicle regions have291 positive/6889 negative patches. These equal-budget linear teacher readouts are not a student-W, future-task, planning or significance result. Source: `reports/st_world_policy/history_data/HANDOFF.md` and its pinned runtime receipts. No new teacher/feature computation is needed for this implementation.

## Fixed comparison

Reuse the actual S-P2 full-training registration recipe (`SP2_fulltrain_seed42_official_5k_v1`) and its existing legal training manifest: current-only, no history/future/delta, W48/view,96 sampled positions/view, teacher FP32, the same PCA basis and application scope, original student preprocessing and S0, original DiT repeat8/10-step deployment. Keep seed42, global32, micro4/rank, world4/accumulation2, original20k cosine horizon/warmup1000, stop5000, Adam settings and backend. Both arms start from the same original S0 with fresh optimizers, not from the already trained S-P2 endpoint. Distinct run names/output namespace prevent mutation of old artifacts.

Allowed config differences are exactly projected `channels`, their derived `transform_identity`, and output directory. A separate study binding carries the common Base/data/PCA pins and arm IDs; no study-only fields are injected into the strict training schema. The pre-existing effective-config validator must parse both arms. No new trainer, decoder, provider, approval bypass or data/caching platform is warranted.

## Risks and interpretation

- PCA is lossy. Its width does not change teacher input pixels, native spatial grid, Qwen visual tokens, W count or deployment compute. Projected target memory is smaller, but raw1024 teacher caches and teacher-tower FLOPs are not reduced.
- Keep the original mean-over-valid-elements loss. Width changes its coordinates and denominator; do not compensate by silently retuning loss weight. Log valid counts and interpret this as a channel-supervision recipe comparison, not strict numerical equivalence.
- The output projection has a different shape; compare shared initialization separately. No checkpoint conversion between widths or warm-start of the trained384 auxiliary head is claimed.
- The64/16 probe dev set has already informed384 selection. It is not an untouched confirmation set. Formal matched evaluation and learned-W diagnostics remain pending; no Navtest-based selection is authorized.
- Source/config/runtime registration and resource reviews must be refreshed for actual runs. Old384 approval does not approve256. Reuse applicable existing evidence without declaring new real integration results.

## Preparation acceptance and deferred execution

Preparation: real JSON configs with exact allowed differences, actual existing PCA metadata and teacher identity references, rejection of wrong width/basis/grid/W/sampling/budget, unchanged original files, and existing config parsing. Small CPU fixtures only. No model/checkpoint/DB/image reads, GPU dispatch or large cache writes.

Actual execution remains Lead-owned and requires explicit new registrations plus independent evidence/correctness/resource review. The implementation will provide pending commands using the existing trainer; it will not fabricate registration or approval artifacts, run training, or claim student gains.

## Lead review and reuse refinement, before configuration implementation

Lead explicitly accepted this preparation after reading this card, retaining independent launch review. Lead also requested checking whether the384 control can be reused rather than rerun. Inspection found the original clean immutable runtime `/mnt/project/VLA-Drive-st-official-runtime-v2-20261016`, commit `fdf25f99b702754787904215bf1c2a2b7515ea75`, already supports both widths. The new256 arm therefore targets that same runtime; the committed384 config is **byte-identical** to the existing S-P2 config. The control's source, Base binding, PCA scope, train manifest, seed, topology, optimizer, initialization and5000-step checkpoint receipt are preserved in `INPUT_PINS.json` and `STUDY_BINDING.json`. The new256 output is isolated. No automatic384 rerun is proposed.

This refines the earlier sentence about two new run identities: only256 needs a new run;384 remains the existing original run, not a copied or relabeled checkpoint. Its native receipt records159992 scene presentations at5000 steps (the original tail batch), not a manufactured160000. This preparation verifies small metadata; it does not reread37GB of checkpoint payload or the full data manifest. Existing validation evidence is reused within its stated scope. Fresh actual256 registration and resource review are still required.

CPU tests establish that the unchanged original initialization context restores outer RNG and every shared sparse-readout parameter is identical between widths. The final output bias is **not** a matching prefix: it is drawn after weight tensors of different sizes. This is disclosed, not silently changed. It does not perturb earlier layers or the restored training RNG. Full initial backbone loading was not rerun; the original canonical FP32 initialization machinery and Base are unchanged.
