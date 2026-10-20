# Verified-pair IEG implementation evidence

Method was fixed in `5852c47b` before this protocol's formal training. Worktree
starts at parent `72e3091a`, retaining native receipt validation and shared
optimizer/resume fixes. The read-only G1 effect-embedding API is from
`3fe37f7c`, applied here as `d5a8c119`. This author does not independently
approve their own mathematics; a different agent reviews it.

All three arms now allocate the same optional effect modules at the end of
adapter initialization. Actual default D is read from the compact native label
schema (currently 11), not hardcoded into a tensor. The original velocity-only
parameters and their seeded initial values remain unchanged. Initialization
uses the existing CPU-only `module_initialization`: no CUDA seed operation,
external CPU RNG consumption or default-device dependency. Legacy `effect_dim=0`
keeps its parameter names and strict state load. Controls allocate identical
extra heads but do not train those heads on effect labels; this distinction
between allocated capacity and active supervision is explicit.

The B1-effect velocity path and target are unchanged from B1-trajectory. Its
noisy-state hidden mean additionally predicts native physical delta and a
projection aligned with frozen G1's antisymmetric real-pair features. True
delta and teacher representation are loss targets only. Frozen G1 is queried
once per repair scene batch, before repeat8; it receives actual saved physical
Base/target endpoints. No predicted endpoint, raw encode/decode roundtrip,
future teacher input to the student, or route-A model is used. The bridge
detaches both teacher outputs. Masks/counts are global across the optimizer
batch and explicit repeat8. Microbatch size changes neither RNG assignment nor
the objective. Native teacher-delta fit error and actual query counts are logged.

The loss remains the preregistered velocity + keep + 0.2 native delta MSE +
0.2 globally gap-weighted direction + 0.2 representation MSE. Gaps are never
cancelled per scene. Empty valid tasks give graph-connected zeros. New training
identities declare `effect_protocol=verified_pair_v1`. Legacy endpoint-proxy
code stays diagnostic and is not silently selected by the new CLI.

`velocity_export` removes both training-only modules and sets effect_dim=0.
The export writer uses that payload. Full Euler10 output is bitwise unchanged
by stripping these modules on the bounded production-class CPU fixture. Hooks
prove neither projection nor effect head is called during deployment. No G1,
refiner, simulator or full preliminary rollout is added.

The 27 current CPU tests pass in 16.41s. New coverage includes three-arm equal
initial state/parameter count, original velocity initialization, CUDA-seed and
default-device isolation, label-independent predictions, private-only gradients,
verified physical teacher arguments, one teacher query before repeat8,
microbatch equivalence, exact gap-cost ratio, simulated DDP/global reductions,
zero valid masks, strict stripped export/full Euler10, and continuation with
both Flow RNG streams. Existing legacy math/trainer tests are also included.

The first integration attempt had 2 failed/25 passed: the CUDA-seed negative
fixture accidentally intercepted the separate tiny Base-head fixture's seed,
and the old parser-only fixture did not isolate the subsequently added native
feedback validator. Both test scopes were corrected explicitly. Production
native validation was retained; it was not bypassed to satisfy a real run.

All tests here use small production modules and labelled synthetic CPU data.
No pretrained policy training, G1 qualification, GPU, simulation or DEV score
was executed by this agent. Auxiliary losses and exact code execution do not
prove useful physical knowledge transfer. Real same-population Euler10/NAVSIM
comparisons, safety preservation and source/log uncertainty remain required.
