# Cost-sensitive variable-K objective

For each scene, use all valid unordered pairs P_x, then a_xij = non_tie*abs(gap)/max(P_x,1). The comparison sampler draws q proportional to a; the sampled softplus loss is not multiplied by gap again. No scene-wise division by sum(abs(gap)) is performed. K=2 therefore retains the0.95/0.02 relative cost. Exact duplicate trajectories are excluded from extra comparison information. K=1/all-tie has zero ranking denominator.

Current tests cover analytic cost ratio, variable-K pair denominators, invalid/tie/NaN padding, stage transition sampler recovery and microbatch equality. A two-process CPU/Gloo test with unequal rank-valid populations matches the single-process global objective gradients within2e-7. Natural pointwise value/component losses are reduced within each scene then across scenes, independently of ranking.

Generated 2026-10-14T07:26:31.328995+00:00. Implementation `0e0ed8d2964c5215d7fcfa712e3f5ab1121ab24d`. Base step 70000, SHA256 `30517484efcc8740c331f42f3f2ba2d1896a4f95cf70e2bf662e0d6c4484e8b0`. All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.
