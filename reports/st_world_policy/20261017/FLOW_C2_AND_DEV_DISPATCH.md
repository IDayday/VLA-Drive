# C2 launch and immutable development endpoints

These are registered ST campaign runs, independent of the bounded future-label
work. They do not restart the held IQE/MOPD experiments or run Navtest.

## Actual C2 launch

C2 was launched on authorized rl-zt2 GPUs4–7 using immutable runtime
`6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`. It retains the common originalS0
initialization, full101592-scene training manifest, global batch32, original
FM repeat8, four10-step rollout candidates,50 CFM samples,4096-update schedule
and100-update warmup. Only the predeclared road-violation reward differs fromC1.
No teacher tensor cache was generated or required.

Registration under `/mnt/project/st-world-policy-20261016/`:
`formal/consequence/registrations/C2_fulltrain_seed42_official_4096_v1/REGISTRATION.json`
SHA256 `7d3dd8587aa9e3f2ee03d113eac9e0535eeadab27598da584f1dfe5a4bda572a`.
Actual invocation: `invocations/1792207212018578877/LAUNCH.json` beside it.

Actual independent review hashes:

- Evidence: `e729663ca910fded8c9b77273864182f0a75af8fdce032f4387f47815c0ed1fe`.
- Correctness: `6c06d263e91980c908478b671209d2ddb5e077692784510c92f6ac9cbcaae5a3`.
- Efficiency: `3f83888a8a03207f4fe7e9de9bb1d27df1175f19cbab349ba608c6d91f7bf9c2`.

The efficiency transfer records the different driver versions explicitly. It
inherits the same integrated runtime exercised on rl-zt2 at world8 and the
actual paired world4 timing on Tsinghua4; it does not claim measured rl-zt2
world4 throughput or cross-driver bitwise equivalence.

Independent live startup review at03:25:03 UTC on2026-10-17 observed4 completed
updates,128 scenes and512 cumulative official candidate calls. All four rank
processes, cwd, source, configuration and GPU mapping matched. Losses/gradients
were finite and FP32 master updates verified. On the first update, all128 road
rewards were0 and all32 scene groups had zero advantage: BC updated, but this
is not evidence of a nonzero road-policy learning signal or scientific gain.

Startup review: `audits/correctness/FORMAL_C2_STARTUP_REVIEW_v1.json`, SHA256
`dcfab3305c3db13e01112037c98489f1d72624c5e7e5ec00dcdb0d0dd52c9744`.
The reviewer checked initialization/status/native frozen identity and optimizer
membership, not a second full running-parameter rehash.

## Development evaluation actually started

Both use the fixed complete1696-scene/16-log dev population, one trajectory,
one visual/language forward and10 DiT steps. The exporter strictly reconstructs
FP32 masters and checks native checkpoint receipts before model use. No Scorer,
trajectory selection or future teacher labels participate.

| Job | Exact native tag | Actual GPU | Launch state |
|---|---|---|---|
| C0 step512 |`step_000512_1792206493515965061`|rl-zt2 GPU0|Launched;311 valid predictions and0 failures in independent03:25:56 snapshot |
| C0 step1024 |`step_001024_1792207671595855958`|rl-zt2 GPU2|Launched at03:28:58 UTC; final export result pending |

Step512 evaluation registration SHA256:
`76bda1d58fd7e8aff8553a83b6c2c58a860ce7c4e41414cba9f00fa51fd8d775`.
Step1024 evaluation registration SHA256:
`c248efd99a95758705309c6c16a94f8cd4b7b87b5ff96d1e8ae19ae916020766`.
Both are under `formal/evaluation/registrations/<job_id>/` in the campaign root.

Step512 launch-to-first-receipt was approximately528 seconds by recorded/file
timestamps. That includes more than model loading; exact monotonic startup/load
times await the final exporter RESULT. Its311-scene prefix mean image-to-trajectory
time was1.50514 seconds and is not a complete-population latency result.
Cold shared-storage startup cannot be replaced with earlier warm16-scene timing.

Official scoring is **not yet started** at this report snapshot. It requires
complete matching export receipts for all1696 scenes. The prepared official
backend command uses2 CPU workers, BLAS1,1728 calls including bounded retry,
and preserves the locked canonical reference context. Global limits remain
at most4 independent GPU exports and2 simultaneous scoring jobs, not per-plan
copies of that allowance.

The unchanged original ten-job preparation and additive C2/C3 preparation are
in `reports/st_world_policy/20261016/evaluation/`. C3 has actual independent
reviews but is queued until GPUs0–3 are released; no launch/runtime identity
has been invented. C4 metadata was prepared separately, awaits its independent
reviews and the C0 training quartet becoming free. No active job is stopped to
make these queues fit.

Scientific status remains pending the fixed matched endpoints. Intermediate
512/1024 results do not authorize early scientific elimination or a new reward.
