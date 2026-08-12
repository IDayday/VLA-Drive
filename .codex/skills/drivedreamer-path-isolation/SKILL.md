---
name: drivedreamer-path-isolation
description: Enforce portable per-developer path configuration and capability-scoped optional dependencies in DriveDreamer-Policy, especially feature/add-VGGT and descendant branches. Use whenever Codex reviews, edits, runs, tests, trains, infers, evaluates, caches data, adds a teacher/model, changes YAML, or creates a launcher in this repository.
---

# DriveDreamer Path Isolation

Keep code and shared configuration portable across developers, branches,
worktrees, local machines, and DLC containers. Never require a developer to
install models belonging only to another research direction.

## Begin Every Task

1. Run `git branch --show-current` and `git status --short`.
2. Work on `feature/add-VGGT` or its intended descendant unless the user names
   another branch. Do not silently switch branches.
3. Preserve `env.local.sh`, untracked data, caches, logs, weights, and user
   changes. Never copy machine paths from another branch into shared files.
4. Read `env.sh`, `load_env.sh`, `env.local.example.sh`, and the affected
   launcher/config before changing a path interface.

## Enforce Path Precedence

Maintain this precedence:

```text
explicit CLI > one-shot environment > env.local.sh > shared YAML/default
```

- Derive `DRIVEDREAMER_ROOT` from the active checkout's `env.sh`; never accept
  a stale inherited checkout root.
- Put personal mounts, credentials, datasets, weights, checkpoints, caches,
  and output roots only in ignored `env.local.sh`.
- Write personal defaults as `${VAR:-local_default}` so one-shot overrides
  continue to work.
- Keep `env.sh` defaults repository-relative. System-local scratch defaults
  such as `/tmp` or `/dev/shm` are allowed only when configurable.
- Add every new persistent path variable to `env.local.example.sh` with a
  commented placeholder. Never put a real developer path in the example.
- Bind new YAML path fields in `starVLA/path_config.py` when direct Python
  training must support them. Apply environment paths before CLI dotlist merge.
- Treat YAML `/path/to/...` values only as inert placeholders; a runnable entry
  must replace them through environment or CLI configuration.

## Write Portable Entrypoints

For each top-level data, cache, training, inference, or evaluation script:

```bash
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_root/load_env.sh"
```

- Make the script runnable from any working directory.
- Quote every path argument and array element.
- Do not use `pwd` to discover the repository.
- Do not source a nonexistent `scripts/load_env.sh`.
- Accept datalist, checkpoint, cache, metric, prediction, and output paths from
  environment variables or explicit CLI arguments.
- Save outputs only below configured output/cache roots. Add generated local
  directories to `.gitignore`; do not delete existing user artifacts.

## Scope Optional Dependencies by Capability

- Loading `env.sh` must never validate or import every optional model.
- Check a dependency only after its capability is selected.
- Action-only training must not require Wan, PPD, DA3, VGGT, V-JEPA, or other
  unrelated teachers.
- Agent-query training may require its own DINO cache without requiring VGGT.
- VGGT development may require its configured local VGGT repository/checkpoint
  without requiring video/depth world-head weights.
- Full video/depth training and their cache builders may require Wan/PPD/DA3.
- Evaluation should require only its checkpoint/predictions, datalist, dataset,
  devkit, and metric cache.
- Use lazy imports for optional packages. Do not auto-download weights. Raise a
  clear error naming the missing variable/path only when that feature is used.

## Keep Data and Cache Contracts Relocatable

- Resolve paths embedded in processed metadata through configured runtime
  roots; do not assume the preprocessing machine still exists.
- Do not use an absolute pathname alone as cache compatibility evidence. Use
  logical component identity plus content/config/checkpoint hashes; absolute
  paths may be retained only as diagnostics.
- Keep each developer's writable cache/output root separate unless a shared
  cache has an explicit manifest and atomic-write contract.
- When loading a saved training config for inference, replace machine-local
  model and data paths with current environment/CLI values.

## Test Before Declaring Success

Write or update CPU-only path contract tests before implementation. At minimum,
verify:

- repository detection works from another working directory and a path with
  spaces;
- a stale inherited `DRIVEDREAMER_ROOT` cannot redirect execution;
- `env.local.sh` is ignored and loaded;
- one-shot overrides beat local defaults;
- CLI path overrides beat environment and YAML;
- launchers preserve path values as single arguments;
- missing unrelated optional models do not block the selected capability;
- active code contains no developer mount prefixes.

Run the relevant subset of:

```bash
pytest -q tests/test_environment_paths.py
bash -n <changed-shell-files>
python -m compileall -q <changed-python-files>
TRAINING_TOPOLOGY_ONLY=1 bash training.sh
git diff --check
```

Do not start full training or evaluation merely to test path wiring. Report
full model/inference/evaluation checks as `NOT RUN` unless actually executed.

## Report the Result

State the active branch, modified files, effective path precedence, capability
dependencies checked, exact commands run, PASS/FAIL/NOT RUN status, missing
optional assets, and whether changes remain uncommitted. Never describe an
unavailable optional model as a repository failure when the selected branch
does not use it.
