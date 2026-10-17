Parent SHA: `ba69838eb13f1e9c28cbee05bc8051b1592ab78e` on `agent/st-evaluation-20261016` in `/mnt/project/VLA-Drive-st-evaluation-agent-20261016`.

Owned paths in this commit:

- `reports/st_world_policy/20261016/evaluation/PREREGISTRATION_EXECUTION_v4.json`
- `reports/st_world_policy/20261016/evaluation/PREREGISTRATION_EXECUTION_v4.md`
- `reports/st_world_policy/20261016/evaluation/EXECUTION_BINDING_VALIDATION_v4.json`
- `reports/st_world_policy/20261016/evaluation/EXECUTION_BINDING_HANDOFF_v4.md`

Cherry-pick only this execution-binding commit onto the Lead integration branch. Existing v1/v2/v3, analysis CLI, all scientific definitions and all W bindings remain unchanged. The preexisting untracked `evaluation/smoke_dev16_aa/` was retained and is not part of this commit.

The actual first C0/C1 invocation receipts and runtime identities are now hash-bound. C2-C4 remain NOT_REGISTERED/NOT_RUN. No runtime, configuration or approval was modified and no evaluation execution permission was added. External evidence paths identify real shared receipts; neither model/data payloads nor raw process inventories are committed.

Validation was limited to small metadata/hash/schema checks and canonical-definition equality; no test suite, training, GPU, export, scoring or scientific analysis was run. Exact plan SHA256: `7176e78ccc287094e19ed51691d02b8dd4685866ea69bd96eb76a2e58c3d9bf4`; validation SHA256: `778a9cc5c3d9e27f22da8e820085fc2c7105509596fd65e5f4492e9b4b048478`; prose SHA256: `53c8e8f94fd737cbabb14b8303d6c6107bef1de433ee83e82147dffb28e87069`.
