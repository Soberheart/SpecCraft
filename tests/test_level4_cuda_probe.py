"""Synthetic CUDA orchestration checks, no GPU or simulator execution."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import probe_level4_cuda as probe


class CUDAProbeTests(unittest.TestCase):
    def test_requires_PBS_before_driver_access(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True), \
                patch.object(probe, 'cuda_driver') as driver:
            out = Path(d) / 'report.json'
            self.assertEqual(probe.child('driver', out, Path(d), 4), 1)
            driver.assert_not_called()
            self.assertIn('PBS allocation', json.loads(out.read_text(encoding='utf-8'))['error'])

    def test_driver_failure_retains_error_and_allocates_nothing(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'PBS_JOBID': 'synthetic.pbs'}, clear=True), \
                patch.object(probe, 'cuda_driver', return_value={'cuInit_code': 3}) as driver:
            out = Path(d) / 'report.json'
            self.assertEqual(probe.child('torch', out, Path(d), 4), 1)
            self.assertEqual(driver.call_count, 1)
            self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['driver']['cuInit_code'], 3)
            self.assertNotIn('scalar_result', json.loads(out.read_text(encoding='utf-8')))

    def test_ambiguous_visible_devices_block_computation(self):
        row = {'cuInit_code': 0, 'device_count_code': 0, 'device_count': 2}
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'PBS_JOBID': 'synthetic.pbs'}, clear=True), \
                patch.object(probe, 'cuda_driver', return_value=row):
            out = Path(d) / 'report.json'
            self.assertEqual(probe.child('taichi', out, Path(d), 4), 1)
            self.assertIn('ambiguous', json.loads(out.read_text(encoding='utf-8'))['error'])
            self.assertNotIn('scalar_result', json.loads(out.read_text(encoding='utf-8')))

    def test_driver_probe_preserves_nonzero_scheduler_mask(self):
        env = {'PBS_JOBID': 'synthetic.pbs', 'CUDA_VISIBLE_DEVICES': '3', 'TI_VISIBLE_DEVICE': '0'}
        row = {'cuInit_code': 0, 'device_count_code': 0, 'device_count': 1}
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, env, clear=True), \
                patch.object(probe, 'cuda_driver', return_value=row):
            out = Path(d) / 'report.json'
            self.assertEqual(probe.child('driver', out, Path(d), 4), 0)
            report = json.loads(out.read_text(encoding='utf-8'))
            self.assertTrue(report['binding_unchanged'])
            self.assertEqual(report['environment_after']['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(report['new_simulations'], 0)

    def test_torch_uses_logical_zero_without_overwriting_physical_mask(self):
        env = {'PBS_JOBID': 'synthetic.pbs', 'CUDA_VISIBLE_DEVICES': 'GPU-synthetic-allocated'}
        row = {'cuInit_code': 0, 'device_count_code': 0, 'device_count': 1}
        calls = []
        class Scalar:
            def __add__(self, value): return self
            def cpu(self): return self
            def item(self): return 3.0
        fake = NS(__version__='synthetic', version=NS(cuda='synthetic'),
                  cuda=NS(device_count=lambda: 1, get_device_name=lambda n: 'synthetic allocated GPU'),
                  tensor=lambda values, device: calls.append(device) or Scalar())
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, env, clear=True), \
                patch.object(probe, 'cuda_driver', return_value=row), patch.dict('sys.modules', {'torch': fake}):
            out = Path(d) / 'report.json'
            self.assertEqual(probe.child('torch', out, Path(d), 4), 0)
            self.assertEqual(calls, ['cuda:0'])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], env['CUDA_VISIBLE_DEVICES'])

    def test_taichi_requires_CUDA_backend_and_disables_fallback(self):
        row = {'cuInit_code': 0, 'device_count_code': 0, 'device_count': 1}
        calls = []
        class Scalar:
            def __setitem__(self, key, value): self.value = value
            def __getitem__(self, key): return self.value
        fake = NS(cuda='syntheticCUDA', i32='i32', cfg=NS(arch='syntheticCUDA'), __version__=(1, 6, 0),
                  init=lambda **kw: calls.append(kw), reset=lambda: None, field=lambda **kw: Scalar())
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'PBS_JOBID': 'synthetic.pbs'}, clear=True), \
                patch.object(probe, 'cuda_driver', return_value=row), patch.dict('sys.modules', {'taichi': fake}):
            self.assertEqual(probe.child('taichi', Path(d)/'report.json', Path(d), 4), 0)
            self.assertEqual(calls, [{'arch': 'syntheticCUDA', 'device_memory_GB': 4, 'enable_fallback': False}])

    def test_failure_and_timeout_retained_without_stopping_other_diagnostics(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def run(command, **kwargs):
                mode = command[command.index('--mode') + 1]
                calls.append(mode)
                if mode == 'torch': raise subprocess.TimeoutExpired(command, 120)
                if mode == 'driver': return NS(returncode=1)
                probe.write(root/(mode+'.json'), {'status': 'passed', 'binding_unchanged': True})
                return NS(returncode=0)
            rows = probe.run_probes(root/'probe.py', root, root, 4, runner=run)
            self.assertEqual(tuple(calls), probe.PROBES)
            self.assertTrue(rows['torch']['timed_out'])
            self.assertFalse(rows['driver']['passed'])
            self.assertTrue(rows['taichi']['passed'])
            self.assertEqual(len(list(root.glob('*_exit.json'))), 4)
            self.assertEqual(len(list(root.glob('*.log'))), 4)

    def test_exit_zero_without_report_is_not_passed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = probe.run_probes(root/'probe.py', root, root, 4, runner=lambda *a, **kw: NS(returncode=0))
            self.assertTrue(all(not r['passed'] for r in rows.values()))


if __name__ == '__main__':
    unittest.main()
