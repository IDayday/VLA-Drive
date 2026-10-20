Privileged GRPO: runnable-v4 (2026-10-20)
Training entry: trainer/README.html and trainer/RUN_GUIDE.html.
Confirmed: 11 arms x 3 epochs x seeds 20261020,20261021,20261022.
Total pipeline deadline: 24h, preserved on recovery. This is a cap, not a throughput guarantee.
History frozen; clean history + 8 action-query suffix; explicit 22 F exceptions <=2 tokens.
28 portable CPU tests pass; real accelerator/multicard/resume acceptance pending.
No formal training or evaluation was started during entry preparation.

From trainer/: copy assets.example.json to assets.local.json and configure external assets.
Generate new prepared_v4 with prepare.py --assets assets.local.json --out prepared_v4
  --monitor-source MONITOR_SOURCE.json --f-policy F_POLICY.confirmed.json.
Then bash launch.sh plan; bash launch.sh check --deep; bash launch.sh run.
Model weights, failure-pair manifests, raw metadata, images, metric caches and prepared
scene text are external. No data or weights are included by this code synchronization.
Legacy reference/ scripts and tools/portable.py are separate historical/asset interfaces.
The original DLC standby launcher is unchanged. No automatic cloud allocation or upload.
