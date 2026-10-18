"""One owned CPU reference invocation and three CLI checks; no model/dispatch."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
OWNER = Path('/mnt/project/VLA-Drive-st-consequence-proxy-evidence-20261018')
SOURCE = OWNER / 'reports/st_world_policy/20261018/consequence_proxy_evidence_v1/reference.py'
OFFICIAL = Path('/mnt/project/st-world-policy-20261016/audits/evidence/public_sources/dfql/agents/dfql.py')
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '479238d0a524e3c1cde786ea327003dbb4d5b275389be6b6e75b70c7cba5cb47'
env = dict(os.environ, PYTHONPATH='/tmp/st-dfql-reference-deps-20261018',
           CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
began = time.monotonic()
results = []

def invoke(name, source, output, expected_code, expected_error=None, help_only=False):
    command = [PYTHON, '-B', str(SOURCE)] + (['--help'] if help_only else
               ['--official-source', str(source), '--output', str(output)])
    out, err = HERE / (name + '.stdout.txt'), HERE / (name + '.stderr.txt')
    with out.open('x') as stdout, err.open('x') as stderr:
        proc = subprocess.Popen(command, cwd=OWNER, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        peak = 0
        while proc.poll() is None:
            status = Path('/proc') / str(proc.pid) / 'status'
            try:
                text = status.read_text()
                for line in text.splitlines():
                    if line.startswith('VmRSS:'): peak = max(peak, int(line.split()[1]) * 1024)
            except FileNotFoundError:
                pass
            if peak > 2 << 30 or time.monotonic() - began > 110 or out.stat().st_size + err.stat().st_size > 1 << 20:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                raise RuntimeError('CPU reference exceeded explicit RSS/time/output budget')
            time.sleep(.05)
    error = err.read_text()
    assert proc.returncode == expected_code, (name, proc.returncode, error)
    assert expected_error is None or expected_error in error
    results.append(dict(name=name, exit_code=proc.returncode, result='PASS', sampled_peak_rss_bytes=peak,
                        command=command, expected_error=expected_error))

output = HERE / 'CPU_REFERENCE_INDEPENDENT.json'
assert not output.exists()
invoke('reference', OFFICIAL, output, 0)
reference = json.loads(output.read_text())
assert reference['test_count'] == 10 and all(x['result'] == 'PASS' for x in reference['tests'])
assert reference['cuda_runtime_initialized'] is False
invoke('help', OFFICIAL, output, 0, help_only=True)
bad = HERE / 'wrong_upstream_fixture.py'
with bad.open('xb') as stream: stream.write(OFFICIAL.read_bytes() + b'\n# invalid source identity fixture\n')
invoke('wrong_upstream_hash', bad, HERE / 'MUST_NOT_EXIST.json', 1, 'Official actor source hash mismatch')
assert not (HERE / 'MUST_NOT_EXIST.json').exists()
before = hashlib.sha256(output.read_bytes()).hexdigest()
invoke('immutable_result', OFFICIAL, output, 1, 'Evidence output is immutable')
assert hashlib.sha256(output.read_bytes()).hexdigest() == before
bad.unlink()  # The only disposable, owned tiny fixture created above.
summary = dict(status='PASS_REFERENCE', arithmetic_checks=10, cli_checks=3, invocations=results,
               elapsed_seconds=time.monotonic()-began, max_rss_limit_bytes=2 << 30,
               max_time_seconds=120, output_limit_bytes=1 << 20, cuda_visible_devices='',
               environment_modified=False, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               model_or_navigation_executed=False, formal_approval=False)
with (HERE/'CHECKS.json').open('x') as f: json.dump(summary, f, indent=2); f.write('\n')
print(json.dumps(summary))
