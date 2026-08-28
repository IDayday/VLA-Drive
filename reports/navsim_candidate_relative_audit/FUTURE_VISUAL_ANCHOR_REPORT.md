# GT Future Visual Anchor Audit

- Scenes audited: **12**
- Future `cam_f0` file coverage: **100.000%**
- Same-frame GT image/pose/annotation/traffic-light/track/structural-target coverage: **100.000%**
- Visual anchor figures written: **12**

## Supported

`I_GT(t+h) <-> C_GT,h`: the front-camera image and GT candidate-relative structured future are resolved from the same logged-future timestamp.

## Not supported

The log does not provide a candidate-specific ground-truth future image for any non-GT candidate. Candidate-conditioned relabeling changes structured relationships, not the recorded pixels.

This audit validates a GT-only visual anchor. It does not train a world model or describe replayed candidate futures as real counterfactual images.
