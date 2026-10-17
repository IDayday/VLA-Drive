"""Independent affected-path CPU checks, not a full Base or launch approval."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
LEAD = Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
PCA = Path('/mnt/project/st-world-policy-20261016/audits/history_data/pca_fixed384_full_current_v1')
SOURCE = 'fdf25f99b702754787904215bf1c2a2b7515ea75'
CONFIG_HASH = 'f71f1cc013701e1f271aad5dffc399f4162c46fabdd1ee896691e4a0f2bcd35e'
started = time.monotonic()
signal.alarm(120)
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
os.sched_setaffinity(0, {2, 3})
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''


def counters():
    values = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
    return {k: int(v) for k, v in values.items()}


def guard():
    while True:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        counts = counters()
        if rss > 2 * 1024**3 or counts['rchar'] > 256 * 1024**2:
            (HERE/'CPU_BUDGET_STOP.json').write_text(json.dumps({
                'status':'BUDGET_STOP','io':counts,'max_rss_bytes':rss,
                'wall_seconds':time.monotonic()-started})+'\n')
            os._exit(90 if rss > 2 * 1024**3 else 91)
        time.sleep(.1)


threading.Thread(target=guard, daemon=True).start()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME, text=True).strip() == SOURCE
assert not subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=RUNTIME, text=True).strip()
sys.path.insert(0, str(RUNTIME))
import torch
torch.set_num_threads(2)
from tools.st_world_policy.training_config import validate_config
from tools.st_world_policy.teacher_pca import TeacherPCA
from starVLA.dataloader.st_world_schema import TeacherFeatureContract
from starVLA.dataloader.st_world_dataset import projected_teacher_contract
from starVLA.model.modules.st_world.sparse_readout import SparseFeatureReadout
from starVLA.model.modules.st_world.losses import masked_mse
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed

details = {}


class AffectedPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = LEAD / 'configs/st_world_policy/teacher_channel_control_v1/PCA256.json'
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == CONFIG_HASH
        cls.cfg = json.loads(content)
        cls.st = validate_config(cls.cfg)

    def test_actual_pca256_basis_and_dataset_contract(self):
        scope = json.loads((PCA / 'CHANNEL_SCOPE.json').read_text())
        entry = next(x for x in scope['allowed_caches'] if x['cache_root'] == self.cfg['data']['teacher_root'])
        p = {}
        for width in (256, 384):
            # Full native384 namespace was already verified for unchanged SP2.
            # Rechecking it here reads its 146MB identity twice per load, outside
            # this focused budget. Keep original raw-contract validation and
            # actual statistics hashes; do not claim a fresh namespace audit.
            p[width] = TeacherPCA.load(PCA, dimension=width, expected_identity=scope['pca_identity'],
                expected_teacher_contract=entry['teacher_contract_identity'])
        raw = TeacherFeatureContract.from_dict(entry['contract'])
        self.assertEqual(projected_teacher_contract(raw, pca_identity=scope['pca_identity'], dimension=256), self.st.teacher)
        self.assertTrue(torch.equal(p[256].tensors['components'], p[384].tensors['components']))
        # Explicit CPU fixture; no teacher feature payload or private scene read.
        x = torch.linspace(-1, 1, 4*1024).reshape(4, 1024)
        a, b = p[256].transform(x), p[384].transform(x)
        torch.testing.assert_close(a, b[:, :256], rtol=1e-5, atol=1e-5)
        details['pca_prefix_max_abs_error'] = float((a-b[:, :256]).abs().max())
        details['pca_statistic_bytes'] = (PCA/'statistics.safetensors').stat().st_size
        details['pca_identity'] = p[256].identity
        details['full_target_namespace_validation'] = 'INHERITED_ORIGINAL_SP2_NOT_REEXECUTED'

    def test_real256_shape_backward_and_strict_head_roundtrip(self):
        torch.manual_seed(99)
        outer = torch.get_rng_state().clone()
        with initialization_seed(42+3600):
            head = SparseFeatureReadout(2048, 256)
        self.assertTrue(torch.equal(outer, torch.get_rng_state()))
        with initialization_seed(42+3600):
            control = SparseFeatureReadout(2048, 384)
        for name, value in head.state_dict().items():
            if not name.startswith('output.'):
                self.assertTrue(torch.equal(value, control.state_dict()[name]), name)
        del control
        memory = torch.randn(1, 144, 2048, requires_grad=True)
        views = torch.arange(3).repeat_interleave(96).unsqueeze(0)
        xy = torch.linspace(-1, 1, 576).reshape(1, 288, 2)
        seconds = torch.zeros(1, 288)
        output = head(memory, views, xy, seconds)
        self.assertEqual(tuple(output.shape), (1, 288, 256))
        loss = masked_mse(output, torch.ones_like(output), torch.ones(1, 288, dtype=torch.bool), 288*256).loss
        loss.backward()
        self.assertTrue(torch.isfinite(memory.grad).all())
        self.assertGreater(float(memory.grad.norm()), 0)
        for name, value in head.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
        details['head_parameter_count'] = sum(p.numel() for p in head.parameters())
        details['fixture_loss'] = float(loss)
        details['fixture_W_gradient_norm'] = float(memory.grad.norm())
        buffer = io.BytesIO()
        torch.save(head.state_dict(), buffer)
        with initialization_seed(43+3600):
            restored = SparseFeatureReadout(2048, 256)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True, map_location='cpu'), strict=True)
        with torch.no_grad():
            actual = restored(memory.detach(), views, xy, seconds)
        torch.testing.assert_close(output.detach(), actual, rtol=0, atol=0)

    def test256_global_denominator_padding_and_empty_mask(self):
        p = torch.full((3, 256), 2., requires_grad=True)
        target = torch.zeros_like(p)
        valid = torch.tensor([True, False, True])
        loss = masked_mse(p, target, valid, 512, world_size=1).loss
        self.assertEqual(float(loss), 4.)
        loss.backward()
        self.assertTrue(torch.equal(p.grad[1], torch.zeros(256)))
        self.assertEqual(float(p.grad[0,0]), 4/512)
        pieces = sum(masked_mse(p[i:i+1], target[i:i+1], valid[i:i+1], 512, world_size=1).loss for i in range(3))
        torch.testing.assert_close(pieces, loss, rtol=0, atol=0)
        empty = masked_mse(p, target, torch.zeros(3, dtype=torch.bool), 0, world_size=1).loss
        self.assertEqual(float(empty), 0.)
        with self.assertRaises(ValueError):
            masked_mse(p, target, valid, 256, world_size=1)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AffectedPathTests))
    metrics = {'schema':'st_channel_affected_cpu_review_v1','tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'skips':len(result.skipped),'wall_seconds':time.monotonic()-started,
        'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'io':counters(),'cpu_affinity':sorted(os.sched_getaffinity(0)), 'runtime':str(RUNTIME),
        'source_sha':SOURCE,'config_sha256':CONFIG_HASH,'gpu_used':False,'real_images_read':False,
        'teacher_payloads_read':False,'full_base_loaded':False,'formal_launch_allowed':False,'details':details}
    (HERE/'CPU_RESULT.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    print(json.dumps(metrics,sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
