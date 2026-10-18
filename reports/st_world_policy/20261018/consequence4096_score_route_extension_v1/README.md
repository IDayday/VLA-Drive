# Fixed DEV CPU score route: authorized Tsinghua zt4

Status: implemented and locally tested; independent review and Lead admission are required before execution. No scoring, inference, training, remote command, or process control was performed by this implementation agent.

Origin: `bec3889c3e6d088afd0076c61b4a35bafa1870dc`, isolated branch `fix/st-score-zt4-route-20261018`. Owned paths are this directory only. The original v5 launcher and all model/trainer/scoring source remain unchanged.

The complete original launcher is loaded only if its SHA256 is `2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3`. The adapter adds exactly `training-vlawm-tsinghua-zt4` to its existing `ROUTE_HOSTS_V3` tuple through an exact AST assignment check. Restoring that tuple makes the entire AST equal to the original, including every function and `main` body. It neither amends old route files nor creates another execution lease.

The actual fresh policy and campaign authorization both allow this canonical host. The adapter still requires the runtime hostname/aliases to match the selected immutable route, live policy, and authorization; prohibited aliases and other hosts are rejected. This is a host routing change, not a scientific adaptation. The error text in the original canonical-host guard still says “three-host”; it is deliberately unchanged because the only adapted AST node is the tuple.

The original formal source SHA, evaluation population, inference, reference score protocol, metadata/payload identity validation, environment equality, expired-route refusal, CPU limits, global shared flock slot limits, duplicate detection, and already-complete-export requirements are inherited. Each scoring invocation still permits two workers, 1728 maximum calls including the manual resume allowance, 3600 native seconds, 4200 outer seconds plus 60 termination grace; no GPU context or automatic retry/follow-up is introduced. The original two invocation maximum, durable admission before Popen, rank lock held through scoring, and existing exporter process check are unchanged. Actual shared mount/CPU/free space eligibility remains a launch-time guard and was not asserted by these tests.

`__file__` is intentionally the thin adapter path so the actual score LAUNCH receipt records the adapter SHA256 `9432aa45d66512e8189e1cdfeb94361fb848e3e226644842e67d87c2d560e5af`. The parent original launcher SHA is pinned in that same file.

## Tests

`python -B reports/st_world_policy/20261018/consequence4096_score_route_extension_v1/test_score_route.py`: 9 tests passed. These cover full AST identity except the single tuple, wrong source and unexpected tuple rejection, actual zt4 policy/authorization and forbidden-alias handling, arbitrary/withdrawn host refusal, original route boundary/expiry checks, and duplicate/invocation budget semantics. The actual C1/C4 route hashes and fixed worker/call budgets are read as small metadata only. No real trajectory, metric cache, checkpoint payload, or model is read.

The actual entry `--help` passed under `/root/miniconda3/envs/ddp/bin/python`; it imports existing software but exits through argparse before runtime validation or dispatch. Test evidence is `TEST_RESULTS.json`. These results do not substitute for real export completeness or independent review.

## Lead-only commands after export completion and independent review

Run on `training-vlawm-tsinghua-zt4-worker-0`, using the existing environment and existing route windows. Each guard requires at least 4260 seconds remaining; an expired route cannot be bypassed by this helper. If the original route becomes insufficient, do not change it or fabricate now; return to Lead for an explicitly reviewed separate execution admission.

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B \
 /mnt/project/VLA-Drive-st-score-zt4-binding-20261018/reports/st_world_policy/20261018/consequence4096_score_route_extension_v1/score_once.py \
 --job-id C1_step004096_dev1696_seed42_v1 \
 --route-sha256 f5507b9cb2a96ac728b555c6d66f5ebb335cd8f93e23f6391b8783ceb6bc8647

PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B \
 /mnt/project/VLA-Drive-st-score-zt4-binding-20261018/reports/st_world_policy/20261018/consequence4096_score_route_extension_v1/score_once.py \
 --job-id C4_step004096_dev1696_seed42_v1 \
 --route-sha256 53e27c1f49c94b05ef44230ba16dac644339d419f4998dcddf8b1a6f7bc1a099
```

Explicit `--resume` is available only through the unchanged original bounded same-identity resume rules, never an automatic loop. The helper still supports only the original predeclared endpoints with valid immutable registrations; adding this host does not create new candidate endpoints.
