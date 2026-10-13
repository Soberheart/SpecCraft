import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scripts.probe_level4_initialization import Recorder, compare_states


class FakeScene:
    def __init__(self):
        self.t = 0
        self.steps = 0

    def step(self):
        self.steps += 1
        self.t += 1

    def reset(self, state=None):
        self.t = 0

    def get_state(self):
        mpm = SimpleNamespace(pos=np.full((2, 3), self.steps, dtype=float), vel=np.ones((2, 3)),
                              C=np.zeros((2, 3, 3)), F=np.broadcast_to(np.eye(3), (2, 3, 3)),
                              active=np.ones(2, dtype=int), mat_id=np.zeros(2, dtype=int))
        tool = SimpleNamespace(pos=np.zeros(3), quat=np.array([1, 0, 0, 0]), vel=np.zeros(3), ang=np.zeros(3))
        return SimpleNamespace(mpm_solver=mpm, tool_solver=SimpleNamespace(data=[tool]), s_global=self.t)


class FakeWorkspace:
    def __init__(self):
        self.scene = FakeScene()
        self.cfg = SimpleNamespace(warmup_horizon=4)
        self.logs = []

    def reset(self):
        for _ in range(4):
            self.scene.step()
        self.scene.reset()

    def log(self, step):
        self.logs.append((step, self.scene.steps))

    def run(self):
        self.reset()
        for step in range(3):
            self.scene.step()
            self.log(step)


class InitializationProbeTests(unittest.TestCase):
    def test_recorder_preserves_step_reset_and_log_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, instrumented = FakeWorkspace(), FakeWorkspace()
            recorder = Recorder(directory)
            recorder.attach(instrumented)
            baseline.run()
            instrumented.run()
            self.assertEqual(baseline.logs, instrumented.logs)
            self.assertEqual(baseline.scene.steps, instrumented.scene.steps)
            self.assertEqual(baseline.scene.t, instrumented.scene.t)
            stages = {r['stage'] for r in recorder.rows}
            self.assertTrue({'constructed', 'warmup_0001', 'warmup_0004', 'scene_reset_01_before',
                             'scene_reset_01_after', 'reset_returned', 'logged_frame0'} <= stages)
            with np.load(Path(directory) / 'logged_frame0.npz', allow_pickle=False) as packet:
                self.assertTrue({'mpm_pos', 'mpm_vel', 'mpm_C', 'mpm_F', 'mpm_active', 'mpm_mat_id',
                                 'tool_000_pos', 'tool_000_quat', 'tool_000_ang'} <= set(packet.files))
                self.assertEqual(packet['mpm_pos'][0, 0], 5)

    def test_saved_state_collision_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(directory)
            workspace = FakeWorkspace()
            recorder.save(workspace, 'constructed')
            before = (Path(directory) / 'constructed.npz').read_bytes()
            with self.assertRaises(FileExistsError):
                recorder.save(workspace, 'constructed')
            self.assertEqual(before, (Path(directory) / 'constructed.npz').read_bytes())

    def test_comparison_preserves_field_units_and_checks_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.npz', Path(directory) / 'b.npz'
            np.savez(a, mpm_pos=np.zeros((2, 3)), mpm_F=np.eye(3))
            np.savez(b, mpm_pos=np.full((2, 3), .00001), mpm_F=np.eye(3))
            report = {r['field']: r for r in compare_states(a, b)}
            self.assertAlmostEqual(report['mpm_pos']['max_abs'], .002)
            self.assertEqual(report['mpm_pos']['unit'], 'mm')
            self.assertTrue(report['mpm_F']['exactly_equal'])
            np.savez(b, mpm_pos=np.zeros((3, 3)), mpm_F=np.eye(3))
            with self.assertRaises(ValueError):
                compare_states(a, b)

    def test_nonfinite_state_never_certifies_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.npz'
            np.savez(path, mpm_pos=np.array([[np.nan, 0, 0]]))
            with self.assertRaises(ValueError):
                compare_states(path, path)


if __name__ == '__main__':
    unittest.main()
