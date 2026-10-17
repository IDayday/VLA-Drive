# Teacher detail versus student learnability: bounded research decision

Date: 2026-10-17 UTC. Lead integration before this document: `44600d38`.
User steering: preserve teacher image detail; reduce unnecessary target complexity;
form a falsifiable hypothesis before another experiment. This is a research
decision, **not an established improvement or authorization to modify active runs**.

## 1. Proposed method and the actual implementation boundary

Keep the teacher's 384×288 input and native 18×24 patch grid. Compress the
1024 channels at each position with the existing fixed training-only PCA basis.
Compare 256 against the already completed, otherwise matched 384-dimensional
S-P2 endpoint. Do not pool the spatial grid, change student input pixels, add W
tokens, enlarge the readout or refit PCA in this comparison.

The current implementation already has channel standardization, non-whitened
PCA and a two-layer, width-512 cross-attention readout. These are reused, not
claimed as new contributions. The new question is whether the extra 128 target
coordinates improve learned information enough to justify their prediction cost.

Hypothesis H1: a compact target preserves relevant teacher information while
reducing student approximation error or harmful shared-backbone updates. A lower
dimension is not inherently better. Low-variance coordinates can still encode
small objects or road boundaries. PCA variance retention alone cannot establish
planning sufficiency.

The existing MSE averages over valid positions **and channels**. More channels
do not automatically increase the loss or gradient. In particular, the shared
first 256 residuals have 1.5 times the coefficient in mean-256 as in mean-384.
This coefficient change is part of the original width recipe and must be
reported; a width result alone cannot isolate capacity from loss weighting.

## 2. Measure the tradeoff explicitly

Let `z` denote the original teacher feature after the fixed training-channel
standardization, `c` the PCA mean, and `U_d` an orthonormal retained basis. For
student prediction `y_hat` in PCA coordinates:

```text
||z - (c + U_d y_hat)||²
  = ||(I - U_d U_d^T)(z - c)||²
  + ||U_d^T(z - c) - y_hat||².
```

The first term is information discarded by target compression. The second is
student prediction error in the retained subspace. This orthogonal decomposition
holds in the fixed standardized coordinates; inverse per-channel scaling does
not preserve the same Euclidean identity in the original raw coordinates.
When averaging over the original 1024 coordinates, the second contribution is
`(d/1024) * MSE_d`, not `MSE_d` directly. Prediction error can contain
representation, optimization and future-observation ambiguity; it is not a
measurement of the student's best achievable error.

Use four separate checks on fixed populations:

1. Teacher information: same-capacity small-object and depth probes on both
   compressed teacher targets; retain reconstruction error as another metric.
2. Student approximation: both models' errors on the same first 256 coordinates,
   plus the 384 model's final 128 coordinates. Match scene/view/patch requests.
3. Student representation: the same probe protocol on frozen W **before** the
   auxiliary prediction head. A stronger head must not substitute for better W.
   Check mismatched or masked W as a read-only control: the readout's view/time/
   position query alone could explain some low error through scene templates.
4. Planning: fixed full-development single-trajectory PDMS, components and
   repaired/new zero-score scenes. Do not choose from Navtest.

The existing 64-train/16-dev teacher probe is only a small diagnostic, not proof
of general equivalence. If information survives PCA but learned W does not
improve, investigate capacity/optimization; if W improves but planning does not,
investigate how planning uses it rather than further minimizing feature loss.

## 3. Next loss candidate, conditional on residual evidence

DINO-Foresight uses PCA targets and SmoothL1(beta=0.1), but its loss ablation
reports broadly comparable results to MSE. Its predictor, teacher and coordinate
scales differ from ours. See the fixed paper/code evidence in `EVIDENCE_CARDS.md`.

Hypothesis H2: a small subset of large feature residuals dominates auxiliary
updates without providing useful planning information. First inspect residual
quantiles and their contributions on a fixed training batch. If supported,
compare the existing MSE to the following **project adaptation**, at one fixed
PCA width and unchanged auxiliary weight:

```text
rho_beta(e) = 2 beta SmoothL1_beta(e)
            = e²                         (|e| < beta)
            = 2 beta |e| - beta²         (otherwise).
```

This preserves MSE's near-zero curvature and bounds the derivative with respect
to each residual. It does not bound the complete parameter gradient through an
arbitrary network Jacobian, equalize total gradient magnitude or remove task
conflict. Select and record beta from the fixed training-coordinate/residual
scale before development evaluation; do not blindly transfer 0.1 across target
normalizations. Report the fraction in the linear regime.

Large residuals can be genuinely important small objects or boundaries. Check
those populations where legitimate labels exist. If the effect is merely less
auxiliary gradient, do not claim a robust-loss mechanism without distinguishing
a simple loss-weight reduction. No robust-loss training was launched here.

## 4. Gradient intervention is conditional, not the default package

Measure FM and auxiliary gradient norms/dot products on parameters genuinely
shared by both objectives, using the existing calibration machinery. Do not use
the gathered final W tensor as a proxy for every shared causal-language path.
Record common-256 and extra-128 contributions with the original 384 denominator.

The more directly relevant main/auxiliary formulation is
[Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/html/1812.02224v2),
Section 3, Equations 3–6 and Algorithm 1. It modulates the auxiliary contribution
to **shared** parameters using its alignment with the main gradient, while
task-specific heads receive their own gradients. Its binary-gate experiments
and positive-cosine weighted formulation are distinct variants. Section 6 and
Appendices E/F discuss local, noisy, high-dimensional gradient limitations.
This is a conditional research option, not a claim that our Adam/ZeRO runtime
inherits the paper's small-step theoretical result. Paper mathematics inspected;
an author-maintained implementation SHA is **not bound**, and no implementation
or formal training approval is asserted for this option.

[GradNorm](https://proceedings.mlr.press/v80/chen18a.html) adapts weights using
gradient norms and relative training rates. A slowly learned auxiliary task may
receive more weight; it is not automatically a remedy for an unpredictable target.

[PCGrad](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
projects conflicting task gradients. Prioritizing FM and projecting only the
auxiliary gradient would be our asymmetric adaptation, not an unchanged PCGrad
implementation. Euclidean first-order alignment does not guarantee improvement
with Adam preconditioning, momentum, finite steps or unseen driving scenarios.
Do not silently install either algorithm before establishing the relevant failure.

## 5. Choices deliberately deferred

- Bigger predictors: existing readout is already nonlinear; lower head loss may
  decouple from student representation quality. This concern is supported by
  [the projector study](https://arxiv.org/html/2303.11098v3), not a proof that
  every larger head fails.
- PHI-S: useful distribution-balancing precedent, but an orthogonal rotation of
  the same error leaves full-channel MSE unchanged. It is not information
  compression or a generic solution to our single-teacher problem.
- Whitening every PCA coordinate or unit-normalizing temporal deltas: changes
  target geometry and can amplify low-energy directions; not included here.
- New teachers, extra student tokens and simultaneous loss/architecture changes:
  not needed to identify this particular tradeoff.

## 6. Execution state

Evidence review: complete paper/code records with fixed source SHAs.
PCA pair preparation and diagnostic code: author tests 9/9 passed; independent
static review `e524057f` found no blocking issue for merging preparation artifacts
(not formal execution approval). A second non-author reviewer checked the
orthogonal decomposition, MSE denominator and robust-loss mathematics. Existing
384 results are reused, never relabeled as a
new run. The new 256 student result is **NOT_RUN**. The existing historical 384
selection remains immutable. New formal training still uses the campaign's
narrow correctness and efficiency approval for the actual registered identity.

Active C-line jobs, temporal implementation and the road-proxy branch remain
independent. No active training recipe, teacher cache, checkpoint or deployment
identity was changed by this research decision. No new GPU experiment was
started to support this document.
