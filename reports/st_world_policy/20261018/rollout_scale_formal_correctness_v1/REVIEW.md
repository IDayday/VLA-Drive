# Independent implementation binding for the sole formal rollout-scale arm

**Correctness role: PASS** for registration SHA `928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f`, exact source `751fcfa2b41549ae16f7b3204aef8dd18b912016`. This is one independent role in the existing three-role approval mechanism. It does not replace evidence/efficiency approval, the complete 240-update PROFILE, resource availability, or the Lead's launch decision. No job was started by this reviewer.

The production `verify_registration` was invoked successfully against the clean frozen runtime. Its exact returned fingerprint is copied into `CORRECTNESS.json`, not reconstructed manually. It verified all seven registered artifact hashes and the committed source fingerprint. Total streaming hash reads were 176,467,158 bytes, including the 176,439,631-byte TRAIN manifest; wall time 1.21 seconds and CPU 0.22 seconds. The manifest was only hashed, not expanded or used to load images, GT, metric caches or model weights. Measured max RSS was 33992 KiB.

## Matched control and initialization

The old C1 and new registered executions are identical: seed42, global batch32, microbatch4, four ranks, full manifest, 4096 optimizer updates and the original LR horizon. Config comparison finds exactly two changed fields: the explicit advantage method and the explanatory adaptation-reason text. All optimizer parameters, warmup100, BC/policy weights1, G4/MC50, repeat8 supervision, precision, saved-boundary interval and fixed scoring protocol remain matched.

The original C1 registration independently confirms identical content hashes for Base binding, the full TRAIN manifest, environment and no-teacher manifest. The Base binding resolves to the same original learned S0, with binding identity `3b910a6d…`; its original master tensors initialize a new matched optimizer. SMOKE/PROFILE outputs are separate diagnostic identities and are not used as initialization. No existing Base checkpoint or completed C1 control is rewritten.

The existing C1 control may therefore be reused without rerunning it solely because the new unused scale branch was added. This does not imply equal source hashes, authorize resuming the old run under a new identity, or establish quality gain.

## Inherited implementation and actual execution evidence

The affected implementation was independently tested at its exact predecessor: 37 CPU/Gloo tests passed, including genuine two-process collectives, bad-input failure votes, tail/unequal partitions, global/microbatch reductions, and tiny DiT/Adam interrupted continuation. The author's separate 97-test evidence is retained. The only subsequent source delta adds nested timing; complete AST equality after removing its four timing assignments was verified.

The actual four-rank SMOKE then completed four updates with 512 valid scored candidates. All 16 rank payloads were checked; one fixed CUDA-compatible FP32 reduction tree exactly reproduced saved raw hashes and normalized FP32 outputs. No tolerance was widened. All-rank common population is128, recorded losses/gradients are finite, FP32 masters update, and the runtime frozen encoder is unchanged.

The latest native checkpoint was checksum-verified once (11.47 GB), and production consumed-normalization guards passed for all ranks without consuming CPU RNG. The new common hash is checked before native RNG restoration, and normal native boundary/terminal behavior is retained. **No new actual GPU/ZeRO resume was executed.** CPU exact-resume and the mature native schema/guard are separate evidence; this report does not silently upgrade them into a GPU resume test. Existing runtime recovery evidence is not invalidated by the unchanged backend.

## Interpretation and outstanding independent role

This variant changes cross-update policy-to-BC weighting. It is a new explicit training definition, not a lossless backend optimization or a guarantee of safety. The scale does not rotate a single-batch policy gradient or make ratio clipping a post-update trust region. Planning benefit remains untested.

Correctness binding is ready. The separate efficiency/resource reviewer still needs the complete actual 240-update window and current resource conditions before the three-role formal approval can authorize launch. This document neither signs that role nor schedules training.
