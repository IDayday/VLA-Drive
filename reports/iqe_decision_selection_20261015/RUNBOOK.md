# Fixed-K4 decision selection: actual commands

Repository: `/mnt/project/DriveVLA-M0-iqe-decision-selection-20261015`.
Branch: `research/iqe-cost-sensitive-selection-1375ad2`.
Audit parent: `1375ad2f67ed841f44bc7e6ec42bcc0f6d301bf0`.
Config: `configs/iqe/decision_selection/main.yaml`.
Output: `/mnt/project/iqe-cost-sensitive-selection-20261015/main_v1`.

The fixed Base is the complete own Query checkpoint
`/mnt/project/iqe-remote-rolling-cache-20261012/continuation/step_070000.pt`,
SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`.
The three frozen expert endpoints and exact existing bank/metric hashes are in
`SELECTION_STUDY_BINDING.json`. No command refreshes Base or the candidate pool.

Commands below use the actual installed environment. Execute on a currently
allowed host after checking `/mnt/project/server_dispatch_policy.json`, live
processes and GPU capacity. The recorded training host is `training-rl-zt2`;
G2 on `training-vla-zt2` remains protected. Withdrawn temporary hosts must not
receive work. Do not repeat an active run or start the held old studies.

```bash
cd /mnt/project/DriveVLA-M0-iqe-decision-selection-20261015
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/envs/ddp/bin/python
CFG=configs/iqe/decision_selection/main.yaml
```

`--help` is available at the module and each subcommand. JSON errors return
nonzero. All stages bind resolved configuration and existing artifact checksums.
The scientific budget is fixed: two 3000-update diagnostics, two single full-D_S
refits, two independent 300-update overfit checks, and the triggered 2000-update
current-state diagnostic. No new expert, optimizer search, Navtest or ACTIVE.

```bash
$PY -m iqe.decision_selection.cli --help
$PY -m iqe.decision_selection.cli bind --config "$CFG"
$PY -m iqe.decision_selection.cli split-inner --config "$CFG"
$PY -m iqe.decision_selection.cli audit-existing --config "$CFG"
```

The internal manifests are fixed at 13139 fit scenes / 169 logs and 3442
validation scenes / 42 logs. Parent D_S contains 16581 scenes / 211 logs. It is
legal incremental-fit data, already seen by Base and R00, not used to fit the
experts. Candidate rewards do not determine the log partition. Complete
official scores and all four candidate identities are checked; missing scores
cannot become zeros. R00 internal-val results are in-sample background.

These two commands can run concurrently on different devices. Global batch128,
microbatch16, warmup100 then constant1e-4, AdamW wd.01, clip1, dropout0; same
natural scene stream. Both run all six internal validation points.

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m iqe.decision_selection.cli train-diagnostic --config "$CFG" --kind M_scene --device cuda:0
CUDA_VISIBLE_DEVICES=1 $PY -m iqe.decision_selection.cli train-diagnostic --config "$CFG" --kind M_set --device cuda:0
CUDA_VISIBLE_DEVICES=2 $PY -m iqe.decision_selection.cli train-overfit --config "$CFG" --kind M_scene --device cuda:0
CUDA_VISIBLE_DEVICES=3 $PY -m iqe.decision_selection.cli train-overfit --config "$CFG" --kind M_set --device cuda:0
```

Overfit runs are separate fixed 64-scene fits and never initialize main models.
After both diagnostics finish, lock the refit plan. Actual internal endpoints
are M_scene2000 and M_set500. The predefined exposure conversion gives2524 and
631 full-D_S updates respectively. These are not extra checkpoint searches.

```bash
$PY -m iqe.decision_selection.cli lock-refit-plan --config "$CFG"
CUDA_VISIBLE_DEVICES=0 $PY -m iqe.decision_selection.cli train-refit --config "$CFG" --kind M_scene --device cuda:0
CUDA_VISIBLE_DEVICES=1 $PY -m iqe.decision_selection.cli train-refit --config "$CFG" --kind M_set --device cuda:0
CUDA_VISIBLE_DEVICES=2 $PY -m iqe.decision_selection.cli run-current-state-probe --config "$CFG" --device cuda:0 --workers 4
```

The two refits each reconstruct the original random initialization, never their
internal best weights. Their last registered endpoint is the only final model.
The conditional probe uses2048 internal-fit and512 internal-val scenes, four
bounded CPU readers, one GPU and2000 updates. It reads current timestamp-attested
agents and static geometry, rejects future-only objects in the official t0
container, and never reads the metric reference as input. It remains privileged
and cannot be exported. It is not a third formal model.

After both refits are complete, run these in order. External stage_val checks
1024 scenes, calibration1024, exploratory dev1696; all are existing fixed
manifests. No role selects a new checkpoint. The margin is a **logit** difference,
not a PDMS difference. Always-Base is an explicit feasible reference, not a
successful learned policy. Raw argmax is always retained.

```bash
CUDA_VISIBLE_DEVICES=3 $PY -m iqe.decision_selection.cli evaluate-external --config "$CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=3 $PY -m iqe.decision_selection.cli calibrate --config "$CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=3 $PY -m iqe.decision_selection.cli evaluate-dev --config "$CFG" --device cuda:0
$PY -m iqe.decision_selection.cli export-research-bundle --config "$CFG" --kind M_scene
$PY -m iqe.decision_selection.cli export-research-bundle --config "$CFG" --kind M_set
```

Real-image and independent process checks use the actual70k Base and trained
heads. They replay the attested original context-extraction batches for numeric
comparison. The benchmark separately uses batch1, the same100 real images per
mode and10 warmups, online image preprocessing and CPU trajectory output.

```bash
CUDA_VISIBLE_DEVICES=4 $PY -m iqe.decision_selection.online --config "$CFG" --device cuda:0
CUDA_VISIBLE_DEVICES=4 $PY -m iqe.decision_selection.online --config "$CFG" --device cuda:0 --fresh-process M_scene
CUDA_VISIBLE_DEVICES=5 $PY -m iqe.decision_selection.online --config "$CFG" --device cuda:0 --fresh-process M_set
CUDA_VISIBLE_DEVICES=6 $PY -m iqe.decision_selection.cli benchmark --config "$CFG" --device cuda:0 --samples 100 --warmup 10
$PY -m iqe.decision_selection.cli audit-process-labels --config "$CFG"
```

Only after all evidence exists, assemble the final report; it refuses to label
missing training, evaluation, real-image or fresh-process checks complete.

```bash
$PY -m iqe.decision_selection.reporting --config "$CFG" --output reports/iqe_decision_selection_20261015
```

One-command execution with the same fixed budgets is available. Use only for a
new isolated study or explicit continuation, never concurrently with the manual
stages above:

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m iqe.decision_selection.cli run-study --config "$CFG" --device cuda:0 --resume
```

## Actual recovery paths

Recovery is for interruption, not an extra optimization stage. The native
checkpoint retains model, Adam, scheduler, RNG, natural sampler consumed cursor,
initialization/bank/split/Z identities and terminal status. Terminal recovery
performs zero additional updates. The actual complete diagnostic checkpoints
created in this run are:

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m iqe.decision_selection.cli resume --config "$CFG" --kind M_scene --phase diagnostic --device cuda:0 --resume /mnt/project/iqe-cost-sensitive-selection-20261015/main_v1/diagnostic/M_scene/step_003000.pt
CUDA_VISIBLE_DEVICES=1 $PY -m iqe.decision_selection.cli resume --config "$CFG" --kind M_set --phase diagnostic --device cuda:0 --resume /mnt/project/iqe-cost-sensitive-selection-20261015/main_v1/diagnostic/M_set/step_003000.pt
CUDA_VISIBLE_DEVICES=0 $PY -m iqe.decision_selection.cli resume --config "$CFG" --kind M_scene --phase refit --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=1 $PY -m iqe.decision_selection.cli resume --config "$CFG" --kind M_set --phase refit --device cuda:0 --resume auto
CUDA_VISIBLE_DEVICES=2 $PY -m iqe.decision_selection.cli run-current-state-probe --config "$CFG" --device cuda:0 --workers 4 --resume auto
```

`auto` selects the latest complete compatible receipt in that phase/model
namespace, not another experiment or a stale result pointer. A started run may
use `--stop-after N` to checkpoint at an optimizer boundary; N does not replace
the declared scheduler horizon. Do not send signals to unrelated jobs. Preserve
failed logs/partial immutable current-state records and rerun the same worker
count to reuse completed records. An incomplete probe never receives COMPLETE.

Research bundle locations:
`.../main_v1/research_bundle/M_scene/bundle.pt` and
`.../main_v1/research_bundle/M_set/bundle.pt`. Each embeds its own trained head
and the fixed expert states, with one strictly verified Base dependency. Load
with `iqe.decision_selection.bundle.load(path, device)`. No GT, metric cache,
privileged current-state probe, external scoring weights or ACTIVE dependency.

For a repeated seed-controlled experiment, copy the config to a new named YAML
and change `experiment_id` and `output_root` before binding. Existing immutable
identities must not be overwritten. This particular study intentionally rejects
a different Base hash/pool: changing Base is a new research registration with
new banks and contexts, not a `latest` switch in this run. No command pushes or
merges code automatically.
