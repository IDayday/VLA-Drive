"""Finite B1 deployment benchmark; import the sealed scientist via PYTHONPATH.

This file deliberately does not add its checkout to sys.path. Main timings call
the existing model.predict unchanged. No labels, scorer, R queries or training.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(16 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def stats(values):
    values = sorted(values)
    def quantile(p):
        at = (len(values) - 1) * p
        lo = int(at)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)
    return dict(n=len(values), mean=statistics.mean(values), p50=quantile(.5),
                p95=quantile(.95), min=values[0], max=values[-1])


def gpu_sample(uuid):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total,memory.free,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    row = next([v.strip() for v in line.split(',')] for line in lines if line.startswith(uuid))
    owners = [line.strip() for line in subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
              if uuid in line]
    return dict(zip(('uuid', 'name', 'driver', 'total_MiB', 'free_MiB', 'temperature_C',
                     'sm_MHz', 'memory_MHz', 'power_W', 'utilization_percent'), row)), owners


def assert_exclusive(uuid, *, initialized):
    row, owners = gpu_sample(uuid)
    if len(owners) != int(initialized):
        raise RuntimeError('Exclusive GPU timing contaminated: ' + repr(owners))
    return dict(gpu=row, owner_rows=owners, at_unix=time.time())


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    assert_exclusive(cfg['gpu_uuid'], initialized=False)
    import torch
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.rl.simwam_recipe.contracts import seed_for
    from starVLA.rl.qwen_world_postcot.data import WorldSceneData
    from starVLA.rl.qwen_world_postcot.batching import to_device
    from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma, clean_x_step
    from starVLA.rl.qwen_world_compare import deployment as compare_deployment
    from starVLA.rl.qwen_world_postcot import deployment as world_deployment

    configure_numerics()
    torch.set_num_threads(cfg['cpu_threads'])
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    source = Path(compare_deployment.__file__).resolve().parents[3]
    if source != Path(cfg['scientist_source']).resolve():
        raise RuntimeError('Imported a different scientist checkout')
    identity = dict(config_sha256=sha(args.config), driver_sha256=sha(__file__),
                    scientist_source=str(source), scientist_commit=subprocess.check_output(
                        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                    host=socket.gethostname(), pid=os.getpid(), torch=torch.__version__,
                    CUDA=torch.version.cuda, cpu_threads=torch.get_num_threads(),
                    started_unix=time.time(), config=cfg)
    write(root/'IDENTITY.json', identity)
    data = WorldSceneData(cfg['data'], role=cfg['data_role'])
    processor_maps = [json.loads((Path(v['export'])/'export.json').read_text())['processor'] for v in cfg['models']]
    if any(v != processor_maps[0] for v in processor_maps):
        raise ValueError('The five official processors differ')
    processor = QwenObservationProcessor(str(Path(cfg['models'][0]['export'])/'processor'))
    manifest = json.loads(Path(cfg['scene_manifest']).read_text())
    if manifest['data_manifest_sha256'] != data.sha:
        raise ValueError('Changed current-observation manifest')
    scenes = manifest['scenes']
    # Stage decoded current frames and processor tensors once. Resident-input
    # measurements reuse tensors, NEVER model hidden, scene memory or prefix.
    prepared = []
    for s in scenes:
        start = time.perf_counter()
        obs = data.observation(s['token'])
        jpeg_ms = 1000*(time.perf_counter() - start)
        cpu = dict(processor([obs]))
        seed = seed_for('native_qwen_sourceflow_ODE', 2101, s['token'])
        noise = torch.randn((1, 8, 3), generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
        prepared.append((s, obs, cpu, noise, jpeg_ms))
    all_results = []
    for item in cfg['models']:
        name = item['name']
        start = time.perf_counter()
        loader = world_deployment.load_export if name == 'Qwen-World' else compare_deployment.load_export
        model, meta = loader(item['export'], device='cuda')
        torch.cuda.synchronize()
        loading_seconds = time.perf_counter() - start
        if meta.get('rl_lora') or meta['qualification']:
            raise ValueError('Architecture comparison requires formal IL deployments')
        # World export predates runtime_source_files: check its model files
        # against the parent scientific receipt instead of disabling checks.
        source_files = meta.get('runtime_source_files')
        if name == 'Qwen-World':
            source_files = {k:v for k,v in meta['IL_parent']['identity']['source_files'].items()
                            if k.startswith('starVLA/model/modules/')}
        for path, digest in source_files.items():
            if sha(source/path) != digest:
                raise RuntimeError('Changed deployment source: '+path)
        model.eval().requires_grad_(False)
        counts = dict(parameters=sum(v.numel() for v in model.parameters()),
                      parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters()))
        if any(v.dtype != torch.float32 for v in model.parameters()):
            raise ValueError('Parameter precision differs from the deployment contract')
        processor_warmup = dict(processor([prepared[0][1]]))
        cur = to_device(processor_warmup, device)
        ego = torch.tensor([prepared[0][1]['proprio']], device=device, dtype=torch.float32)
        noise = prepared[0][3].to(device)
        with torch.no_grad():
            for _ in range(cfg['warmup']):
                model.predict(cur, ego, noise)
            torch.cuda.synchronize()
        rows, phase_rows, hardware = [], [], []
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for s, obs, cpu, root_noise, jpeg_ms in prepared:
                hardware.append(assert_exclusive(cfg['gpu_uuid'], initialized=True))
                cur = to_device(cpu, device)
                ego = torch.tensor([obs['proprio']], device=device, dtype=torch.float32)
                noise = root_noise.to(device)
                torch.cuda.synchronize()
                for repeat in range(cfg['repeats']):
                    # Primary 1: exact existing predict, GPU inputs resident.
                    start = time.perf_counter()
                    reference = model.predict(cur, ego, noise)
                    torch.cuda.synchronize()
                    model_ms = 1000*(time.perf_counter()-start)
                    # Primary 2: already decoded RGB -> CPU physical trajectory.
                    # No synchronization is inserted between its sub-phases.
                    start = time.perf_counter()
                    fresh_cpu = dict(processor([obs]))
                    fresh = to_device(fresh_cpu, device)
                    fresh_ego = torch.tensor([obs['proprio']], device=device, dtype=torch.float32)
                    fresh_noise = root_noise.to(device)
                    out = model.predict(fresh, fresh_ego, fresh_noise)
                    trajectory = out['requested_trajectory'].cpu()
                    torch.cuda.synchronize()
                    rgb_ms = 1000*(time.perf_counter()-start)
                    if not torch.equal(out['normalized_action'], reference['normalized_action']):
                        raise AssertionError('Official processor replay changed the native prediction')
                    rows.append(dict(token=s['token'], physical_log=s['physical_log'], repeat=repeat,
                                     model_ms=model_ms, rgb_to_trajectory_ms=rgb_ms,
                                     one_time_file_decode_ms=jpeg_ms,
                                     total_tokens=int(cpu['attention_mask'].sum()),
                                     image_grid_thw=cpu['image_grid_thw'].tolist()))
                # Secondary diagnostic, separate pass with phase boundaries.
                # Verify output against primary; never report this as end-to-end.
                phase = dict(token=s['token'])
                start = time.perf_counter(); processor([obs])
                phase['processor_ms'] = 1000*(time.perf_counter()-start)
                torch.cuda.synchronize(); start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if name == 'Qwen-World':
                        context = model.encode_condition(cur, ego)
                    else:
                        context = model.encode_current(cur, ego)
                    torch.cuda.synchronize()
                    phase['encode_current_ms'] = 1000*(time.perf_counter()-start)
                    start = time.perf_counter()
                    prefix = model.actor.prepare_context(context) if name == 'Qwen-World' else model.prepare_prefix(context, auxiliary=False)
                    torch.cuda.synchronize()
                    phase['prepare_prefix_ms'] = 1000*(time.perf_counter()-start)
                    start = time.perf_counter()
                    grid = shift_sigma(torch.linspace(1, 0, 11, dtype=torch.float32), 5.).tolist()
                    state = noise.float(); events = []
                    for sigma, nxt in zip(grid[:-1], grid[1:]):
                        time_value = torch.full((1,), sigma, device=device, dtype=torch.float32)
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        prediction = model.actor(state, time_value, prefix) if name == 'Qwen-World' else model.denoise_x0(state, time_value, prefix)
                        end.record(); events.append((begin,end))
                        state = prediction.float() if name == 'Qwen-World' and nxt == 0 else clean_x_step(state, prediction, sigma, nxt, validate=False)
                    if name == 'Qwen-World' and not bool(torch.isfinite(state).all()):
                        raise FloatingPointError('Nonfinite world ODE')
                    torch.cuda.synchronize()
                    phase['K10_ms'] = 1000*(time.perf_counter()-start)
                    phase['denoiser_gpu_ms_by_step'] = [a.elapsed_time(b) for a,b in events]
                parity = (state - reference['normalized_action']).abs().max().item()
                if not torch.equal(state, reference['normalized_action']):
                    raise AssertionError('Diagnostic split path changed native K10 output: '+str(parity))
                phase['normalized_action_max_abs_error'] = parity
                phase['physical_trajectory_max_abs_error'] = (ActionCodec.denorm_odo(state)-reference['requested_trajectory']).abs().max().item()
                phase_rows.append(phase)
                del context, prefix, state, prediction, reference, out, fresh, cur, ego, noise
                write(root/(name+'_PROGRESS.json'), dict(completed_scenes=len(phase_rows), total_scenes=len(prepared), at_unix=time.time()))
        hardware.append(assert_exclusive(cfg['gpu_uuid'], initialized=True))
        result = dict(name=name, IL_updates=meta['IL_updates'], export=item['export'],
                      export_sha256=sha(Path(item['export'])/'export.json'), model_sha256=meta['model_sha256'],
                      runtime_source_files=source_files, loading_seconds=loading_seconds,
                      counts=counts, rows=rows, phases=phase_rows, hardware=hardware,
                      peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                      peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
                      summary={key:stats([r[key] for r in rows]) for key in ('model_ms','rgb_to_trajectory_ms')},
                      phase_summary={key:stats([r[key] for r in phase_rows]) for key in ('processor_ms','encode_current_ms','prepare_prefix_ms','K10_ms')})
        write(root/(name+'.json'),result);all_results.append(result)
        print(json.dumps(dict(name=name, summary=result['summary'], phase_summary=result['phase_summary'])),flush=True)
        del model
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    write(root/'RESULTS.json', dict(identity=identity,results=all_results, finished_unix=time.time()))
    write(root/'COMPLETE',dict(results_sha256=sha(root/'RESULTS.json')))


if __name__ == '__main__':
    main()
