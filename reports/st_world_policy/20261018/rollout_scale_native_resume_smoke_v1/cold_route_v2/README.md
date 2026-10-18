# Cold-process runtime import correction and explicit v2 namespace

The real v1 initial driver failed before START/claim/Popen in route_checks:
fresh_environment_v3 imported tools without runtime751 on sys.path. The original
trace is retained in V1_INITIAL_DRIVER_FAILURE.log. This was a preflight error,
not a timeout; no unchanged retry was made. Inspection found no v1 control
invocations. The v1 prepared configs/binding/lease/logs remain untouched.

The runtime patch is deliberately two local changes: route_checks inserts the
exact frozen RUNTIME before calling mature helpers; ROOT becomes the explicit
`/mnt/project/st-world-policy-20261016/smoke/rollout_scale_native_resume_v2`.
Architecture, data32, native behavior, optimizer, normalization, scopeSMOKE,
4096 schedule/warm100, stop4,8 total updates,1024 calls and zero retries are
unchanged. v1 consumed0 updates/calls; v2 continues that same finite overall
budget. Runtime751 and the formal run are not modified.

Old worktree `/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-20261018`
is restored clean at19a5a244 with its original controller SHA693d9307. Its
reviewers' absolute source pins remain valid. Only this agent's two just-edited
files were restored from their original git bytes, after preserving a patch and
applying it in the new v2 worktree. Own untracked test evidence was moved, not
discarded. WORKTREE_MIGRATION.json and the complete owned patch record this;
there was no reset, stash or user-edit overwrite.

New source worktree:
`/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v2-20261018`.
Branch: fix/st-rollout-native-resume-cold-import-v2-20261018.

Only2 affected cold-process tests are run; original38/independent51 are not
repeated or claimed as a new full suite. Each probe is a fresh isolated Python
process with runtime paths and all tools modules removed. The old exact19a5
source genuinely reproduces ModuleNotFoundError: tools; the corrected source
calls the real frozen fresh_environment_v3 and imports training_config from
exact runtime751. All unrelated route inputs are explicit CPU fixtures, with no
GPU/mount/affinity change or model/score operation. SOURCE_PINS.json identifies
the actual tested source and original failure. Final measurements are in
TEST_EXECUTION.json and complete stdout in TEST_OUTPUT.txt. The earlier run from
the original worktree before migration is retained separately.

Real GPU dispatch is still WAITING_NEW_V2_BINDING_AND_TWO_LOCAL_REVIEWS.
No v2 admission, model process or training was created by this implementer.
