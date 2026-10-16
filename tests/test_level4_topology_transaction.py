import copy
from types import SimpleNamespace as NS
import unittest
import numpy as np
from level4.topology_transaction import PhysicalTransaction, without_final_step


class Field:
    def __init__(self, value): self.value = np.array(value, dtype=float)
    def to_numpy(self): return self.value.copy()
    def from_numpy(self, value): self.value[:] = value


class MPM:
    def __init__(self):
        self.slots = [{key: np.array([i + 1.]) for key in ('pos','vel','C','F','active')}
                      for i in range(8)]
    def get_state(self, f): return copy.deepcopy(self.slots[f])
    def set_state(self, f, state): self.slots[f] = copy.deepcopy(state)


class Sim(NS):
    @property
    def cur_substep_local(self): return self.cur_substep_global % 8


class Scene:
    def __init__(self):
        tool = NS(**{key: Field(np.arange(8)) for key in
                     ('pos','quat','vel','ang','latest_pos','latest_quat')})
        tool._tgt = {'pos': np.array([2.]), 'vel': None}
        tool.update_mesh_pose = lambda f: None
        tool.copy_frame = lambda source,target: [getattr(tool,key).value.__setitem__(target,getattr(tool,key).value[source])
                                                for key in ('pos','quat','vel','ang')]
        material = NS(_tgt={'vel': np.array([3.]), 'col': None})
        self.sim = Sim(cur_substep_global=3, max_substeps_local=8,
                      mpm_solver=MPM(), tool_solver=NS(entities=[tool]), entities=[tool,material])
        self.t = 17
    def get_state(self): return NS(mpm_solver=self.sim.mpm_solver.get_state(self.sim.cur_substep_local))
    def set_state(self, state):
        self.sim.cur_substep_global = 0
        self.sim.mpm_solver.set_state(0, state.mpm_solver)
    def step(self):
        self.sim.cur_substep_global += 1
        self.t += 1


class Topology:
    def __init__(self, scene): self.scene = scene; self.graph = 0
    def check(self):
        self.graph += 1
        self.scene.step()
        self.scene.step()


class TransactionTests(unittest.TestCase):
    def test_restore_physics_pending_and_clock_on_success_and_exception(self):
        for fail in (False, True):
            scene = Scene(); tool = scene.sim.tool_solver.entities[0]
            before = {k: getattr(tool,k).to_numpy() for k in
                      ('pos','quat','vel','ang','latest_pos','latest_quat')}
            material = copy.deepcopy(scene.sim.mpm_solver.slots[2:4])
            def branch():
                with PhysicalTransaction(scene):
                    for field in before: getattr(tool,field).value[:] = -99
                    for slot in scene.sim.mpm_solver.slots:
                        for key in slot: slot[key][:] = -1
                    for entity in scene.sim.entities: entity._tgt = {'vel': np.array([999.])}
                    scene.step()
                    if fail: raise RuntimeError('diagnostic failure')
            if fail:
                with self.assertRaisesRegex(RuntimeError,'diagnostic failure'): branch()
            else: branch()
            self.assertEqual((scene.sim.cur_substep_global,scene.t),(0,17))
            for key, value in before.items():
                if key in ('pos','quat','vel','ang'):
                    value[0] = value[3]
                    value[7] = value[2]
                np.testing.assert_array_equal(getattr(tool,key).to_numpy(),value)
            for a,b in zip(material,scene.sim.mpm_solver.slots[2:4]):
                for key in a: np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(tool._tgt['pos'],[2.])
            self.assertIsNone(tool._tgt['vel'])
            np.testing.assert_array_equal(scene.sim.entities[1]._tgt['vel'],[3.])

    def test_remove_only_final_step_preserve_graph_and_branch(self):
        scene = Scene(); topology = Topology(scene)
        check = without_final_step(topology.check)
        with PhysicalTransaction(scene): check()
        self.assertEqual(topology.graph,1)
        self.assertEqual((scene.sim.cur_substep_global,scene.t),(0,17))

    def test_reject_different_source_shape(self):
        with self.assertRaises(ValueError): without_final_step(Scene().step)

    def test_ring_start(self):
        scene = Scene(); scene.sim.cur_substep_global = 0
        state = scene.get_state().mpm_solver
        with PhysicalTransaction(scene): scene.step()
        for key in state: np.testing.assert_array_equal(scene.sim.mpm_solver.slots[0][key],state[key])

    def test_repeated_branch_rollbacks_do_not_add_physical_time(self):
        scene = Scene(); initial = scene.t
        topology = Topology(scene); check = without_final_step(topology.check)
        for _ in range(63):
            scene.step()
            with PhysicalTransaction(scene): check()
        self.assertEqual(scene.t-initial,63)
        self.assertEqual(topology.graph,63)
        self.assertEqual(scene.sim.cur_substep_global,0)

    def test_rebased_predecessor_matches_latest_at_ring_interior_end_and_start(self):
        for counter in (3,7,8):
            with self.subTest(counter=counter):
                scene=Scene();scene.sim.cur_substep_global=counter
                tool=scene.sim.tool_solver.entities[0]
                previous=(counter-1)%8;current=counter%8
                before={key:getattr(tool,key).to_numpy() for key in ('pos','quat','vel','ang')}
                for key in ('pos','quat'):
                    getattr(tool,'latest_'+key).value[:] = before[key][previous]
                previous_material=scene.sim.mpm_solver.get_state(previous)
                current_material=scene.sim.mpm_solver.get_state(current)
                with PhysicalTransaction(scene):
                    for key in before:getattr(tool,key).value[:] = -100
                    scene.step()
                for key,original in before.items():
                    self.assertEqual(getattr(tool,key).value[0],original[current])
                    self.assertEqual(getattr(tool,key).value[7],original[previous])
                for key in ('pos','quat'):
                    self.assertEqual(getattr(tool,key).value[7],getattr(tool,'latest_'+key).value[0])
                for key in current_material:
                    np.testing.assert_array_equal(scene.sim.mpm_solver.slots[0][key],current_material[key])
                    np.testing.assert_array_equal(scene.sim.mpm_solver.slots[7][key],previous_material[key])


if __name__ == '__main__': unittest.main()
