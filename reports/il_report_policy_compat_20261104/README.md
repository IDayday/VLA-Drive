# IL report recovery and D68k result, 2026-11-04

The model evaluator and training continued correctly. The HTML report failed
because `allowed_hosts` changed from a metadata mapping to an explicit list of
five containers. The old report called `.items()` on that list. This exception
occurred during presentation assembly, after scoring; it did not change model
trajectories, rewards, optimizer updates or completed evaluation files.

The isolated fix supports both receipt shapes. It reads only explicitly listed
hosts, joins current per-host metadata, and honors withdrawal and temporary
expiry. It also includes per-host metadata in inventory/event signatures so a
change to that metadata cannot reuse an obsolete resource review. This helper
does not grant experiment permission or restart a terminated experiment.

## Executed checks and deployment

- `python3 -m unittest discover -s tests/il_report -p 'test_*.py' -q`:
  **59 tests passed**, including the new container-schema checks.
- A real read-only resource probe resolved exactly the five authorized
  containers, twice, with no errors. No other hosts were contacted by this probe.
- Replaced only the CPU HTML owner, using its recorded PID/start identity and
  cooperative SIGTERM. The original HTML and receipts were archived first.
  The native learner and evaluation scheduler received no signals.
- New report owner: PID `2604198`, start ticks `1520803350` on the local
  `training-vlawm-tsinghua-zt3` container. It uses this isolated report checkout
  with the existing task-suite/periodic presentation wrapper.
- Actual HTML rendering succeeded at **2026-11-04 05:13:57 UTC**, with 209
  completed IL results, D68k included, and no report warnings.
- The original 30-minute regular reporting cadence and metadata event updates
  remain enabled. The Navtest checkpoint interval remains **2,000 optimizer
  updates**, with its existing sole scheduler PID `2590878` unchanged.
- The native D learner PID `2582218` continues toward its original 100k target;
  all eight rank journals have matching optimizer counters, finite losses and
  zero skipped optimizer steps. No stopped training was restarted.

Private operation receipts remain in the shared experiment storage under
`qwen-world-il-research-20261030/maintenance/report_policy_compat_20261104/`.
They include the original error, prior HTML/snapshot, exact launch command,
read-only resource inventory, and `VERIFICATION.json`. They are not model or
dataset uploads.

## Newly available full Navtest results

All rows use canonical NAVSIM v1.1, 12,146 scenes / 44 physical logs, invalid=0,
single candidate, B1, clean-x K10 shift5 ODE, seed2101. Numbers are percentage
points, not raw reward units.

| D IL checkpoint | PDMS |
| --- | ---: |
| 56k, previous highest | 89.7409 |
| 60k | 89.2729 |
| 62k | 89.6572 |
| 64k | 89.3774 |
| 66k | 89.4455 |
| **68k, current highest** | **89.7623** |

The 68k improvement over 56k is only **0.0214 PDMS points**. The sequence
fluctuates; this does not establish a sustained improvement or a material
performance breakthrough. These are repeatedly inspected test results, not a
fresh blind test or a multi-training-seed uncertainty estimate.

For D68k, NC=98.7280, DAC=97.4066, TTC=95.7929, EP=83.5398,
Comfort=99.9835, DDC=98.5674. Both model-level and job-level completion receipts
confirm all owned evaluation workers exited. No model inference or simulation
was repeated to repair the HTML.

The complete result was already published separately on
`reports/il-bd-navtest-live-20261103`, commit
`26a7af733def32bee49468404f983e21d9ac5bb7`, with remote SHA verified.
The next scheduled D checkpoints are 70k, 72k, and subsequent 2k boundaries.
