# Legacy endpoint-proxy effect-query abstention

This helper is a diagnostic prerequisite, not approval of the endpoint-proxy
method or a positive scientific result. It does not modify the three primary
training arms. The primary verified-pair representation method is separately
specified before training; it must not silently turn into plain SFT because a
proxy query was unsupported.

`endpoint_support(base_raw, endpoint_raw)` only accepts original FP32 raw8x4
tensors. It decodes them using the original normalizer and fits the physical XY
change to the actual canonical quintic intervention basis. Windows 1 and 5 stay
disabled because G0 did not expose them. Each longitudinal coefficient and
pointwise longitudinal displacement is bounded by 2m; lateral counterparts by
0.6m. FP32 storage/normalizer roundoff tolerance is fixed at 2e-5m/rad. Rebuilt
yaw must agree, relative heading change is additionally bounded by 0.5rad, and
the original declared geometric motion limits must hold. The extra heading
bound is a conservative numeric restriction, not an official NAVSIM metric.
No threshold is fitted or expanded after observing DEV.

Passing means only `NUMERIC_ENVELOPE_AVAILABLE`. It does not prove that a new
combination interpolates the teacher's data, that the controller can execute
it, or that it is safe. Native initial velocity and tracking remain unverified.
Every failure reason and overall applicability fraction are returned. All
unsupported yields `NOT_IDENTIFIABLE`. Nonfinite model output is an error,
never converted to silent abstention or zero score.

Any legacy caller must apply the support mask **before** the teacher forward,
query only supported rows, and report actual coverage. Computing teacher
outputs for arbitrary proxies and masking only the loss is insufficient.
Microbatch/rank losses require global supported numerators/denominators;
averaging local supported means is wrong. Zero supported rows need a
graph-connected zero without a teacher call. This helper does not implement
that integration and cannot be used to claim it has been tested.

Eight analytic CPU tests passed in 3.28s: identity/exposed interventions,
longitudinal/lateral bounds, both unexposed windows, combined coefficients
escaping the physical radius, arbitrary non-basis/noisy proxies, illegal yaw,
nonfinite failures and immutable exposure configuration. They execute no model,
simulator or teacher. Real query coverage has not been measured here.
