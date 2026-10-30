# Qwen-World IL living research report

Live entry: `/mnt/project/qwen-world-il-research-20261030/report.html`.

The HTML opens directly. The reporting program requires **only system Python 3.10 standard libraries**; its runtime does not require Chromium, Playwright, a GPU, model weights, or teachers. Existing RL inotify code is preserved verbatim in `tools/il_report/rl_events_reference.py`; the IL subclass watches several registered experiment roots. `event_reuse.json` binds the copied source hash.

The report contains:

- Source-checked SVG diagrams for the current Qwen-World and A/B/C/D, with training-only versus deployment components, shared weights, exact-candidate replay, and gradient boundaries. SVG download, tab switching, deployment-only view, fullscreen, chart legends, and table filtering use local code.
- All four from-scratch loss pilots and their actual10k screening results, separated from63k→65k objective-switch pilots. The retained methods are TIME_RESIDUAL/HUBER; a selected run with a failed terminal worker is explicitly shown as failed, not silently running.
- Eight-rank common update counts, immutable checkpoint updates, current terminal receipts, local loss/throughput windows, complete canonical Navtest curves and submetrics, separate DEV diagnostics, and explicit next steps.
- A finite, existing-journal-only matched5k analysis: same32-scene order, same500-update windows, no skipped optimizer steps. It does not infer an architecture's final ceiling from train reconstruction loss or shared-hardware step time.
- A separate progress panel for the ten newly registered midpoint evaluations. Complete results join the main table automatically. Partial scene averages are never published.

Updates follow the RL rule: every1800seconds; complete evaluations, meaningful training milestones, important failures, selections, and resource changes can trigger immediate updates. Events do not reset the regular clock. Only metadata is watched; no per-update journal polling occurs. The unique reporting owner holds `reporting/watch.lock`. All output files are atomic; transient render failures preserve the last usable HTML.

```bash
PYTHONPATH=. /usr/bin/python3 -m unittest discover -s tests/il_report -v
/usr/bin/python3 -m tools.il_report.render --once
/usr/bin/python3 -m tools.il_report.render --watch --interval 1800
/usr/bin/python3 -m tools.il_report.diagnose_architectures --output /mnt/project/qwen-world-il-research-20261030/reporting/matched_5k_analysis.json
```

Do not start a second watch owner. The existing report owner is identified by `/mnt/project/qwen-world-il-research-20261030/reporting/OWNER.json` (PID and process start time). This reporter has no training, scheduler, resource-dispatch, scoring or checkpoint-modification function. It does not replace existing campaign owners.

The architecture comparison uses formal scientific source `fdaff809652370328e45c0d996ad706e1af2fc0c`, inspected against the executed model, independent/shared/world/Qwen wrappers and configs. The old baseline has four auxiliary tasks; the new comparison has seven. Their absolute scores are informative but are not a pure architecture-only causal ablation against the historical baseline. Within A/B/C/D, the common task and data contract is held fixed.

Main scores: full12146, canonical NAVSIMv1.1, one clean-x ODE trajectory, K10/shift5, seed2101; all-invalid/partial/wrong-protocol/wrong-seed/changed-seal results are withheld. Test has been observed repeatedly. The fixed DEV parent-selection rule of the four-way100kIL→50kRL campaign is unchanged.
