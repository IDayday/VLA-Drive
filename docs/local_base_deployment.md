# Local DriveVLA-M0 Base/no-memory deployment

This route deploys the paper's **91.0 PDMS Base Model** used in the ablation
tables: InternVL3, 64 learned trajectory proposals, and the learned scorer,
without retrieval, memory, test-time training, trajectory refinement, or the
Scale route. The paper's 92.3 result adds map+agent retrieval and TTT; it is not
the development baseline configured here.

The corresponding NAVSIM target is:

```text
navsim.agents.EpisodeDrive.episodedrive_agent.EpisodeDriveAgent
```

## PPU-safe environment

The host image owns the vendor-adapted runtime. `scripts/setup_local.sh` creates
a virtual environment with `--system-site-packages`, verifies every audited
runtime version, and installs only this repository with `pip --no-deps`. It
does not execute either requirements file and cannot resolve or replace torch,
torchvision, torchaudio, triton, flash-attn, or deepspeed.

The audited PPU build includes `torch==2.4.0`, `triton==3.0.0+ppu1.7.0.oe`,
and `flash-attn==2.8.2+v0.1.0.ppu2.1.0.oe`. These are exact compatibility
contracts; the setup script checks them before and after editable installation.

```bash
./scripts/setup_local.sh
```

If the runtime version check fails, select the audited PPU image. Do not repair
the failure with a generic PyPI or CUDA PyTorch wheel.

## Machine-local paths

Copy `env.local.example.sh` to the ignored `env.local.sh`, then set:

- merged Base checkpoint;
- local InternVL3-2B config/tokenizer directory;
- local DINOv2 safetensors file;
- NAVSIM test logs and sensor blobs;
- nuPlan maps;
- NAVSIM 1.1 navtest metric cache;
- writable output directory.

Configuration precedence is: explicit Hydra CLI override, one-shot exported
environment variable, ignored `env.local.sh`, then repository default. The
repository root is always recomputed from this checkout so an inherited root
from another worktree cannot redirect imports or outputs.

`OPENSCENE_DATA_ROOT` must contain:

```text
navsim_dataset/
|-- meta_datas/test
`-- sensor_blobs/test
```

The loaders are offline by default (`HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`) so an accidental network lookup cannot silently
change the model assets.

## Validation and execution

First validate the exact checkpoint, DINO weights, tokenizer, data, metric
cache, topology, and accelerator:

```bash
./scripts/preflight_base.sh --full-hash
```

For a read-only asset/code check in a shell where the PPU device was not
mounted, use `--allow-no-accelerator`. This does not count as a model smoke
test.

Run one navtest scene end to end:

```bash
./scripts/run_base_smoke.sh
```

Run the complete 12,146-scene evaluation:

```bash
./scripts/run_base_pdms.sh
```

By default, both launchers select the released Base/no-memory topology:
64 proposals,
8 poses over 4 seconds, four scorer references, frozen InternVL initialized
from local config, local DINO weights, and no retrieval route.

## Validation on the PPU host

The deployment was validated on 2026-08-28 with one `PPU-ZW810E` (95.6 GiB):

- full SHA-256 and checkpoint-structure preflight: PASS;
- merged checkpoint load: PASS;
- one-scene VLM forward, 64-trajectory decode, scorer selection: PASS;
- NAVSIM PDMS evaluation: 1/1 scenes succeeded;
- smoke token `7da6ba784b8b5ff0`: PDMS `0.8689785588834177`.

The one-scene score is a functional smoke result, not a reproduction claim for
the paper's 91.0 aggregate. That requires the full navtest command above.
