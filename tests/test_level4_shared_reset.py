import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from level4.shared_reset import load_shared_state
from scripts.probe_level4_initialization import Recorder, attach_recorder, compare_states, collect


def make_state():
    mpm = SimpleNamespace(pos=np.zeros((2, 3), dtype=np.float32), vel=np.zeros((2, 3), dtype=np.float32),
                          C=np.zeros((2, 3, 3), dtype=np.float32), F=np.zeros((2, 3, 3), dtype=np.float32),
                          active=np.ones(2, dtype=np.int32), mat_id=np.zeros(2, dtype=np.int32))
    tool = SimpleNamespace(pos=np.zeros(3, dtype=np.float32), quat=np.array([1, 0, 0, 0], dtype=np.float32),
                           vel=np.zeros(3, dtype=np.float32), ang=np.zeros(3, dtype=np.float32))
    return SimpleNamespace(mpm_solver=mpm, tool_solver=SimpleNamespace(data=[tool]), s_global=99)


def make_packet():
    # Independently specified expected values, including hidden C/F/velocity.
    return dict(mpm_pos=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                mpm_vel=np.full((2, 3), .25, dtype=np.float32), mpm_C=np.full((2, 3, 3), .5, dtype=np.float32),
                mpm_F=np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy(),
                mpm_active=np.array([1, 0], dtype=np.int32), mpm_mat_id=np.zeros(2, dtype=np.int32),
                tool_000_pos=np.ones(3, dtype=np.float32), tool_000_quat=np.array([0, 0, 0, 1], dtype=np.float32),
                tool_000_vel=np.full(3, 2, dtype=np.float32), tool_000_ang=np.full(3, 3, dtype=np.float32))


class SharedResetTests(unittest.TestCase):
    def test_complete_state_restores_hidden_fields_and_clock(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.npz'; packet=make_packet(); np.savez(path, **packet)
            state=make_state(); report=load_shared_state(path, state)
            for key in ('pos','vel','C','F','active'):
                np.testing.assert_array_equal(getattr(state.mpm_solver,key), packet['mpm_'+key])
            for key in ('pos','quat','vel','ang'):
                np.testing.assert_array_equal(getattr(state.tool_solver.data[0],key), packet['tool_000_'+key])
            self.assertEqual(state.s_global,0)
            self.assertTrue(report['mat_id_placeholder_not_used'])
            self.assertTrue(report['static_material_map_requires_frozen_source_and_config'])

    def assert_rejected_without_mutation(self, change):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.npz'; packet=make_packet(); change(packet); np.savez(path, **packet)
            state=make_state(); before=copy.deepcopy(state)
            with self.assertRaises(ValueError): load_shared_state(path, state)
            for key in ('pos','vel','C','F','active','mat_id'):
                np.testing.assert_array_equal(getattr(state.mpm_solver,key),getattr(before.mpm_solver,key))
            for key in ('pos','quat','vel','ang'):
                np.testing.assert_array_equal(getattr(state.tool_solver.data[0],key),getattr(before.tool_solver.data[0],key))
            self.assertEqual(state.s_global,99)

    def test_missing_deformation_history_rejected(self):
        self.assert_rejected_without_mutation(lambda p:p.pop('mpm_F'))

    def test_nonfinite_late_field_cannot_partially_restore(self):
        self.assert_rejected_without_mutation(lambda p:p['tool_000_ang'].__setitem__(0,np.nan))

    def test_dtype_mismatch_rejected(self):
        self.assert_rejected_without_mutation(lambda p:p.update(mpm_pos=p['mpm_pos'].astype(np.float64)))

    def test_shape_mismatch_rejected(self):
        self.assert_rejected_without_mutation(lambda p:p.update(mpm_C=np.zeros((2,3),dtype=np.float32)))

    def test_extra_tool_rejected(self):
        self.assert_rejected_without_mutation(lambda p:p.update(tool_001_pos=np.zeros(3,dtype=np.float32)))

    def test_illegal_or_empty_mask_rejected(self):
        for mask in ([-1,1],[0,0],[1,2]):
            with self.subTest(mask=mask):
                self.assert_rejected_without_mutation(lambda p:p.update(mpm_active=np.array(mask,dtype=np.int32)))

    def test_nonunit_quaternion_rejected(self):
        self.assert_rejected_without_mutation(lambda p:p.update(tool_000_quat=np.ones(4,dtype=np.float32)))

    def test_material_placeholder_cannot_certify_actual_map(self):
        self.assert_rejected_without_mutation(lambda p:p.update(mpm_mat_id=np.ones(2,dtype=np.int32)))

    def test_tensor_like_state_restoration(self):
        class Tensor:
            def __init__(self, value): self.data=value.copy(); self.shape=value.shape; self.dtype='torch.'+str(value.dtype)
            def new_tensor(self, value): return Tensor(value)
            def copy_(self, other): self.data[:]=other.data
        with tempfile.TemporaryDirectory() as d:
            packet=make_packet(); path=Path(d)/'state.npz'; np.savez(path,**packet); state=make_state()
            for key in ('pos','vel','C','F','active'):
                setattr(state.mpm_solver,key,Tensor(getattr(state.mpm_solver,key)))
            load_shared_state(path,state)
            np.testing.assert_array_equal(state.mpm_solver.F.data,packet['mpm_F'])

    def test_shared_mode_keeps_original_setup_and_skips_warmup(self):
        class Scene:
            def __init__(self): self.state=make_state(); self.t=0; self.steps=0
            def get_state(self): return copy.deepcopy(self.state)
            def reset(self, state): self.state=copy.deepcopy(state); self.t=0
            def step(self): self.steps+=1; self.t+=1
        class Workspace:
            def __init__(self):
                self.scene=Scene(); self.init_state=None; self.setup=0; self.logs=[]
            def reset(self):
                self.setup+=1  # Original scene-dir/config/controller setup.
                if self.init_state is None:
                    for _ in range(80): self.scene.step()
                    self.init_state=self.scene.get_state()
                self.scene.reset(self.init_state)
            def log(self,step): self.logs.append(step)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=root/'source.npz'; np.savez(path,**make_packet())
            recorder=Recorder(root); workspace=Workspace(); attach_recorder(recorder,workspace,path)
            workspace.reset(); workspace.scene.step(); workspace.log(0)
            self.assertEqual(workspace.setup,1)
            self.assertEqual(workspace.scene.steps,1)
            self.assertEqual(workspace.logs,[0])
            self.assertTrue(all(r['exactly_equal'] for r in compare_states(path,root/'reset_returned.npz')))

    def test_preflight_failure_is_packaged_and_not_success(self):
        import json
        from level4.archive import verify_zip
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); request=root/'request.json'; request.write_text(json.dumps({'probe_source_sha256':'wrong'}), encoding='utf-8')
            with self.assertRaises(ValueError): collect(request,root/'data')
            result=verify_zip(root/'diagnostic_evidence.zip')
            self.assertEqual(result['status'],'passed')
            failure=json.loads((root/'data/collector_failure.json').read_text(encoding='utf-8'))
            self.assertFalse(failure['candidate_pool_released'])


if __name__=='__main__': unittest.main()
