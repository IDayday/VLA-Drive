"""Observe fixed-chain FP32 chunk differences without changing any gate.

FP64 arithmetic isolates mean sensitivity from probability evaluation roundoff.
This is a read-only diagnostic, not production acceptance or a new sampler.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from starVLA.rl.flow_grpo.comparison import tensor_comparison
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.loading import load_policy
from starVLA.rl.flow_grpo.math import gaussian_logprob, reduce_dimensions
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.acceptance import executable_identity
from starVLA.rl.flow_grpo.rollout import evaluate_transitions
from starVLA.rl.flow_grpo.temporal_noise import whiten


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--behavior', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    configure_numerics()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    cfg, sft = resolve_config(args.config)
    sft.framework.qwenvl.load_device = 'cuda'
    sft.framework.qwenvl.load_dtype = 'float32'
    sft.framework.qwenvl.attn_implementation = 'sdpa'
    policy = load_policy(cfg, sft, SimpleNamespace(process_index=0, device=torch.device('cuda')))
    policy.cuda().bfloat16().float().eval()
    behavior = torch.load(args.behavior, map_location='cuda', weights_only=False)
    traces = []
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        for chunk in (1, 2):
            chain = replace(behavior, spec=replace(behavior.spec, candidate_chunk_size=chunk))
            traces.append(evaluate_transitions(policy, behavior.observation, chain))
    a, b = traces
    assert torch.equal(a['std'], b['std'])
    value = behavior.chain[:, :, 1:].double()
    std = a['std'].double()
    rho = behavior.spec.temporal_noise_correlation
    means = [x['mean'].double() for x in traces]
    exact = [gaussian_logprob(value, mean, std, rho) for mean in means]
    z = whiten((value-means[0])/std, rho)
    dz = whiten((means[1]-means[0])/std, rho)
    analytical_delta = z*dz - .5*dz.square()
    torch.testing.assert_close(exact[1]-exact[0], analytical_delta, atol=1e-11, rtol=1e-9)
    comparison = {k: tensor_comparison(a[k], b[k]) for k in a}
    bad = (b['elementwise_logprob'].double()-a['elementwise_logprob'].double()).abs() > (
        2e-6 + 2e-3*a['elementwise_logprob'].double().abs())
    indices = bad.flatten().nonzero().flatten().tolist()
    samples = []
    for i in indices:
        samples.append({'flat_index': i,
                        'fp32_values': [float(x['elementwise_logprob'].flatten()[i]) for x in traces],
                        'fp64_values': [float(x.flatten()[i]) for x in exact],
                        'analytical_mean_delta': float(analytical_delta.flatten()[i]),
                        'fp32_roundoff': [float((x['elementwise_logprob'].double()-y).flatten()[i])
                                           for x, y in zip(traces, exact)]})
    report = {'scope': 'FP32 fixed-chain chunk 1/2 no-grad explanation; not a gate override',
              'historical_tolerance_status': 'PASS' if not indices else 'FAIL',
              'execution': executable_identity(), 'token': behavior.observation.tokens,
              'device': torch.cuda.get_device_name(), 'dtype': 'float32',
              'attention_backend': 'SDPA MATH', 'tf32': torch.backends.cuda.matmul.allow_tf32,
              'layers': comparison, 'failed_elements': samples,
              'analytic_identity_max_residual': float((exact[1]-exact[0]-analytical_delta).abs().max()),
              'logprob_fp64_max_delta': float((exact[1]-exact[0]).abs().max()),
              'dimension_mean_ratio_max_error': float((
                  (reduce_dimensions(b['elementwise_logprob'])-reduce_dimensions(a['elementwise_logprob'])).exp()-1).abs().max()),
              'probability_roundoff_max_abs_by_chunk': [
                  float((x['elementwise_logprob'].double()-y).abs().max()) for x, y in zip(traces, exact)]}
    torch.save({'traces': [{k: v.cpu() for k, v in x.items()} for x in traces],
                'fp64_logprob': [x.cpu() for x in exact]}, out/'tensors.pt')
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
