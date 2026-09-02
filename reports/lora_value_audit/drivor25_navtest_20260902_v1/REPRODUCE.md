# Reproduce

All commands run in `/mnt/project/DriveVLA-M0-lora-value-audit-20260902` with no network access and no LoRA training.

1. Run `scripts/lora_value_audit/run_f0_parity.sh` to export rank-sharded proposals and verify exact forward parity.
2. Run `scripts/lora_value_audit/run_f1_base_audit.sh` to produce the full Base-64 audit and grouped scorer analysis.
3. Score every external bank with `tools.lora_value_audit.score_candidate_bank`; this uses fixed PDM-reference progress already proven identical to official one-candidate `pdm_score`.
4. Run `scripts/lora_value_audit/run_union_bank.sh` so the frozen scorer sees each complete Base+external set in one self-attention call.
5. Re-run this module with the same arguments to apply `tools.lora_value_audit.schema.choose_verdict`.

The exact realized commands and all absolute artifact paths/hashes are in `commands.log` and `manifest.json`.
