"""Bounded CPU formula audit; no VLA, learned critic, simulator, or trainer.

The upstream actor_loss is extracted unchanged from its SHA-pinned source and
executed with deterministic toy network functions. The PyTorch transcription of
its arithmetic is covered by the accompanying upstream MIT notice.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import resource
from pathlib import Path
import sys
import time

UPSTREAM_SHA = "8b6b15803cc1e5605ad1524553a2c95e103b0bad"
ACTOR_SHA256 = "c893151303f507bd77fdb918b8a956d92a582c9f2f6aa7cb9acbcc132c4ecc21"


def run(source: Path):
    # Explicit CPU setup before either framework import; one process / one CPU.
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    import numpy as np
    import torch
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != ACTOR_SHA256:
        raise ValueError("Official actor source hash mismatch")
    tree = ast.parse(data)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFQLAgent")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "actor_loss")
    # No edit/transformation to the function's body. Only dependencies and the
    # mathematical network are replaced, not its stop-gradient or reductions.
    namespace = {"jax": jax, "jnp": jnp}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    original_actor = namespace["actor_loss"]
    D, K, B = 32, 10, 3
    obs = jnp.asarray([[.1, .2], [.3, -.2], [-.4, .5]], dtype=jnp.float64)
    labels = jnp.linspace(-.4, .6, B * D).reshape(B, D)
    center = jnp.linspace(-.5, .4, D)
    curvature = jnp.linspace(.7, 1.3, D)
    params = {"bias": jnp.linspace(-.08, .09, D),
              "scale": jnp.linspace(-.2, .3, D),
              "time": jnp.linspace(-.15, .2, D)}
    rng = jax.random.PRNGKey(142)
    _, x_rng, _ = jax.random.split(rng, 3)
    x0 = jax.random.normal(x_rng, (B, D))
    evidence = {"schema": "dfql_cpu_formula_reference_v1", "status": "REFERENCE_EXECUTED",
        "formal_launch_allowed": False, "upstream_commit": UPSTREAM_SHA,
        "upstream_actor_source_sha256": ACTOR_SHA256,
        "executed_upstream_symbol": "DFQLAgent.actor_loss (unchanged AST; toy network only)",
        "dimensions": {"batch": B, "raw_coordinates": D, "euler_steps": K},
        "environment": {"torch": torch.__version__, "jax": jax.__version__,
                        "numpy": np.__version__, "python": sys.version,
                        "devices": [str(d) for d in jax.devices()], "dtype": "float64",
                        "cpu_affinity": sorted(os.sched_getaffinity(0))}, "tests": []}

    def check(name, values):
        evidence["tests"].append({"name": name, "result": "PASS", **values})

    def compare(a, b, label, atol=2e-11):
        if not (np.isfinite(np.asarray(a)).all() and np.isfinite(np.asarray(b)).all()):
            raise AssertionError(f"{label}: non-finite reference value")
        error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        if error > atol:
            raise AssertionError(f"{label}: {error} > {atol}")
        return error

    class ToyNetwork:
        def select(self, kind):
            if kind == "actor_bc_flow":
                return lambda o, a, t, params: 5 * jnp.tanh((
                    params["bias"] + params["scale"] * a + params["time"] * t + o[:, :1] * .01) / 5)
            if kind != "critic":
                raise ValueError(kind)
            def critic(o, a=None, actions=None):
                a = actions if actions is not None else a
                q0 = 2 - .5 * jnp.sum(curvature * (a - center) ** 2, axis=-1)
                q1 = 1.5 - .5 * jnp.sum(curvature[::-1] * (a + center * .7) ** 2, axis=-1)
                return jnp.stack([q0, q1])
            return critic

    class ToyAgent:
        network = ToyNetwork()
        def __init__(self, agg, normalize):
            self.config = {"actor_type": "DFQL", "flow_steps": K, "alpha": .7,
                           "q_agg": agg, "normalize_q_loss": normalize}
        def sample_actions(self, observations, seed):
            # Diagnostic MSE only; never contributes to actor_loss.
            return jnp.zeros((len(observations), D))

    def tensor(x, grad=False):
        return torch.tensor(np.asarray(x).copy(), dtype=torch.float64, requires_grad=grad)

    def torch_source(p, agg, normalize):
        o, target, initial = tensor(obs), tensor(labels), tensor(x0)
        a = initial
        vs, bcs = [], []
        def v(state, t):
            return 5 * torch.tanh((p["bias"] + p["scale"] * state + p["time"] * t + o[:, :1] * .01) / 5)
        for k in range(K):
            t = k / K
            vel = v(a, t)
            vs.append(vel)
            bcs.append(v((1-t) * initial + t * target, t))
            a = (a + vel / K).detach()
        clipped = a.clamp(-1, 1).detach().requires_grad_(True)
        c, cur = tensor(center), tensor(curvature)
        qs = torch.stack([2 - .5 * (cur * (clipped-c).square()).sum(-1),
                          1.5 - .5 * (cur.flip(0) * (clipped+c*.7).square()).sum(-1)])
        qagg = qs.mean(0) if agg == "mean" else qs.min(0).values
        g, = torch.autograd.grad(qagg.sum(), clipped)
        weights = 1 / torch.arange(K, 0, -1, dtype=torch.float64)
        q_loss = -2 * .7 / K * (torch.stack(vs) * g.detach()[None] * weights[:,None,None]).mean()
        if normalize:
            q_loss = q_loss / qs.mean(0).abs().mean().detach()
        bc = (torch.stack(bcs) - (target-initial)[None]).square().mean()
        return bc + q_loss, bc, q_loss, a, torch.stack(vs), g.detach()

    for agg in ["mean", "min"]:
        for normalize in [False, True]:
            agent = ToyAgent(agg, normalize)
            (jl, metrics), jg = jax.value_and_grad(original_actor, argnums=2, has_aux=True)(
                agent, {"observations": obs, "actions": labels}, params, rng)
            tp = {k:tensor(v, True) for k,v in params.items()}
            tl, bc, ql, endpoint, velocities, g = torch_source(tp, agg, normalize)
            tg = torch.autograd.grad(tl, list(tp.values()))
            errors = {"loss": compare(jl, tl.detach().numpy(), "loss"),
                      "bc": compare(metrics["bc_flow_loss"], bc.detach().numpy(), "BC"),
                      "q": compare(metrics["q_loss"], ql.detach().numpy(), "q"),
                      "gradient": max(compare(jg[k], v.numpy(), k) for k,v in zip(tp, tg))}
            assert (endpoint.abs() > 1).any(), "Need actual clipping case in toy"
            check(f"unchanged_official_actor_vs_torch_{agg}_normalize_{normalize}", errors)

    # Paper Eq 14/15: at the SAME theta snapshot the shifted-MSE and dot-product
    # gradients agree, while their scalar loss values need not agree.
    v = tensor(np.linspace(-.4,.3,K*B*D).reshape(K,B,D), True)
    g = tensor(np.linspace(-.2,.5,B*D).reshape(B,D))
    w = 1 / torch.arange(K,0,-1,dtype=torch.float64)
    quadratic = (.5 * (v - (v + g[None]/K).detach()).square().sum(-1) * w[:,None]).sum(0).mean()
    linear = -((v * g.detach()[None]).sum(-1) * w[:,None] / K).sum(0).mean()
    gq, = torch.autograd.grad(quadratic, v, retain_graph=True)
    gl, = torch.autograd.grad(linear, v)
    check("paper_shifted_MSE_and_linear_gradient_equivalence", {
        "gradient_max_abs_error": compare(gq.numpy(), gl.numpy(), "Eq14/15"),
        "losses_are_not_identical": float(quadratic) != float(linear)})

    source_q = -2 * .7 / K * (v * g[None] * w[:,None,None]).mean()
    ratio = float(source_q / (.7 * linear))
    compare(ratio, 2/(K*D), "paper/code reduction ratio", 1e-13)
    check("paper_sum_vs_source_mean_scale", {"source_to_alpha_paper_RL_ratio": ratio,
        "expected": 2/(K*D), "paper_BC_to_source_BC_factor": K*D,
        "within_source_vs_paper_RL_to_BC_ratio_factor": 2})

    # Full BPTT reference with constant affine Jacobian, closed-form sensitivity
    # and finite difference. No detached intermediate states in this reference.
    initial = torch.linspace(-.1,.1,D,dtype=torch.float64)
    A = torch.diag(torch.linspace(-.4,.3,D,dtype=torch.float64))
    A[0,1] = .2
    forcing = torch.stack([torch.sin(torch.arange(D,dtype=torch.float64)*.1+k*.17) for k in range(K)])
    reward_grad = torch.linspace(-.3,.7,D,dtype=torch.float64)
    theta = torch.tensor(.13,dtype=torch.float64,requires_grad=True)
    def endpoint(theta):
        a = initial
        for k in range(K):
            a = a + (A @ a + theta * forcing[k]) / K
        return a
    loss = -(endpoint(theta)*reward_grad).sum()
    auto, = torch.autograd.grad(loss, theta)
    tangent = torch.zeros(D,dtype=torch.float64)
    for k in range(K):
        tangent = (torch.eye(D,dtype=torch.float64)+A/K) @ tangent + forcing[k]/K
    analytic = -(reward_grad*tangent).sum()
    epsilon = 1e-6
    fd = float((-(endpoint(theta.detach()+epsilon)*reward_grad).sum() +
                (endpoint(theta.detach()-epsilon)*reward_grad).sum())/(2*epsilon))
    check("full_BPTT_matches_Euler_sensitivity_and_finite_difference", {
        "autograd":float(auto), "analytic":float(analytic), "finite_difference":fd,
        "analytic_error":compare(float(auto),float(analytic),"BPTT analytic"),
        "finite_difference_error":compare(float(auto),fd,"BPTT FD",2e-9)})

    # Finite-K shared-parameter counterexample, no state Jacobian at all.
    # One coordinate among 32; b_0=1, b_9=-.2, otherwise 0; R(q)=q[0].
    # Full R'= .1*(1-.2)=.08; weighted DFQL R'= .1*(.1-.2)=-.01.
    sensitivities = torch.zeros(K,D,dtype=torch.float64)
    sensitivities[0,0], sensitivities[-1,0] = 1., -.2
    theta = torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    a = torch.zeros(D,dtype=torch.float64)
    for k in range(K):
        a = a + theta*sensitivities[k]/K
    bptt_loss = -a[0]
    dfql_loss = -(theta*sensitivities[:,0]*w/K).sum()
    bg, = torch.autograd.grad(bptt_loss, theta, retain_graph=True)
    dg, = torch.autograd.grad(dfql_loss, theta)
    eta = .01
    reward_delta_bptt = .08 * (-eta*float(bg))
    reward_delta_dfql = .08 * (-eta*float(dg))
    assert reward_delta_bptt > 0 and reward_delta_dfql < 0
    check("shared_parameter_finite_K_opposite_direction_counterexample", {
        "state_Jacobian_norm":0., "K":K, "D":D,
        "loss_gradient_BPTT":float(bg), "loss_gradient_DFQL_paper":float(dg),
        "reward_delta_BPTT":reward_delta_bptt,"reward_delta_DFQL":reward_delta_dfql,
        "claim_refuted":"DFQL always shares BPTT direction at K=10 even without exploding Jacobians"})

    # Import ONLY existing raw->physical decoder: no image / metric / VLA load.
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo))
    from starVLA.dataloader.foresight_dataset import decode_ego, X_MEAN, X_STD, Y_MEAN, Y_STD
    raw = torch.tensor([[[.3,-.2,.6,.8]]*8],dtype=torch.float64,requires_grad=True)
    physical = decode_ego(raw)
    cost = (physical[...,0] + 2*physical[...,1] + 3*physical[...,2]).sum()
    grad, = torch.autograd.grad(cost, raw)
    expected = torch.tensor([X_STD,2*Y_STD,2.4,-1.8],dtype=torch.float64).expand_as(raw)
    check("actual_navigation_decoder_raw_chain_rule", {
        "max_abs_error":compare(grad.numpy(),expected.numpy(),"raw decoder",2e-12),
        "raw_shape":[1,8,4], "physical_shape":list(physical.shape),
        "robot_clipping_xy_box_m":[[X_MEAN-X_STD,X_MEAN+X_STD],[Y_MEAN-Y_STD,Y_MEAN+Y_STD]],
        "no_navigation_clipping_used": True})
    # True score inserted as a constant has no reward gradient to the actor.
    t = torch.tensor(.2, dtype=torch.float64,requires_grad=True)
    bc = t.square()
    g0, = torch.autograd.grad(bc, t, retain_graph=True)
    g1, = torch.autograd.grad(bc + torch.tensor(.91), t)
    check("constant_true_score_does_not_create_policy_gradient", {
        "gradient_before":float(g0), "gradient_after":float(g1),
        "equal": bool(torch.equal(g0,g1))})
    evidence["elapsed_seconds"] = time.monotonic()-started
    evidence["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    evidence["cuda_runtime_initialized"] = torch.cuda.is_initialized()
    assert not evidence["cuda_runtime_initialized"]
    evidence["reference_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evidence["test_count"] = len(evidence["tests"])
    evidence["limits"] = ["Toy CPU arithmetic only, not real DiT or navigation",
        "Official actor_loss executed; full Flax agent/training loop not instantiated",
        "No real imagery, official simulator, actual-road proxy, optimizer update, DDP or resume",
        "No inference-time guidance/refiner/selection added", "No formal scientific or implementation approval"]
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("Evidence output is immutable; choose a new attempt path")
    result=run(args.official_source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"tests":result["test_count"],"path":str(args.output)}))

if __name__ == "__main__":
    main()
