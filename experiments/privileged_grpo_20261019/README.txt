Privileged-feedback Action-only DiT GRPO experiment migration

Read RUN_GUIDE.html first. Commands assume this directory is the current directory.

Ready: frozen source snapshot, tokenizer metadata, original model configuration,
1264/141 split identities, historical research implementation references,
portable CPU asset validation, a NEW training-side 8-candidate inference smoke,
and byte-exact weight split/restore and GitHub Release upload tools.

Implemented: trainer/run.py and trainer/schedule.py for 11 arms, scope-specific
updates, paired evaluation and full resume. All 11 arms freeze the history encoder (history-frozen-v2).
12 CPU tests passed; the revised joint scope still needs real-PPU revalidation.
Historical single-PPU smokes are preserved with their original scope definitions.
Still gated: 22 unmatched strict F training donors, training-length pilot,
approved formal budget, and multi-card acceptance.
The DLC launcher currently preflights and waits; it does not train.
Do not interpret archived short probes as completed full experiments.

Data and weight blobs are external; config/assets.example.json defines interfaces.
Raw images, metadata, metric caches, and failure/GT manifests are not published here.
Token/log split identities are preserved for exact reconstruction and validation.
The original constructor also needs tokenizer/model.safetensors, supplied separately
from the complete Base checkpoint in the planned GitHub Release.

reference/ preserves historical scripts as evidence. They retain their original
paths and dependencies on old run artifacts. Do not run them as portable launchers.
tools/portable.py is the new migration entrypoint. Its GPU smoke still requires
execution and acceptance on the destination accelerator before formal computation.

See trainer/README.html and trainer/ACCEPTANCE.json for the new formal entry.
