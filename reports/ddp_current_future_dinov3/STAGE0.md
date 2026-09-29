# Actual implementation and verification

Incremental source from3832ab6. Legacy FLUX files are retained, but semantic W_* configurations use only the new DINO head and label schema. Original DDP ego FM and current Qwen input path are unchanged. One current Qwen forward supplies W and action queries; auxiliary readouts do not feed predictions into the action head. Physical seconds0/1/2/4share one head. Current and requested-future losses have distinct global denominators; missing future labels never resample an easier horizon. Model parameters start from generic Qwen plus independently seeded random driving/head modules.

31focused CPU tests passed. A real generic DINO encoder CPU check passed on a current training image: full16:9 field,16x29patch grid,448fully supported patches,1024channels, deterministic frozen eval and RNG preservation. FP16 MSE3.19e-9, maxabs9.71e-4on this one image; full-cache quantization statistics will be separate. No new GPU or student training claim at this commit.

Full current/future image index completed with0failures; see IMAGE_INDEX.json. It uses the unchanged101592train/1696dev population and1192disjoint logs. h=0was checked against the exact student current image path and timestamp. Raw data stay read-only. Deduplicated image features have independent scene/horizon references. Index extraction ran on CPU from the newly written script; its exact writer hash is in the artifact manifest (the base SHA alone does not contain that new writer).

Prior GT-MAE checkpoint and interaction-cache identities verified, including full split index and input-side ego mask; reused without retraining. Original user DINO improvement recipe remains UNVERIFIED; fallback and source are explicit in RECIPE_RECOVERY.json and ENCODER_CPU.json. No prior student result is inserted into the new comparison.

Existing current-image preprocessing remains original DDP16:9; new encoder refuses a differing field until a common transform is registered. Source images observed so far are1920x1080. Complete cache will reject unexpected geometry, failed current encodings or changed image identities rather than silently drop scenes.

Next executed stage: immutable-source GPU encoding check, full target extraction, real W/Qwen gradients and deployment isolation, calibrated short fits and matched full-data student training. Formal throughput and auxiliary weights are not yet measured for this new configuration.
