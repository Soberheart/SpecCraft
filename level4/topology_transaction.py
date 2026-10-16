"""Prospective forward-only topology diagnostics with physical-state rollback.

This adapter changes execution semantics and does not certify legacy model
alignment. It preserves graph updates but never retains diagnostic integration.
"""
import ast
import copy
import hashlib
import inspect
import textwrap
import types


def without_final_step(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    function = tree.body[0]
    expected = ast.parse('self.scene.step()').body[0]
    if not isinstance(function, ast.FunctionDef) or ast.dump(function.body[-1]) != ast.dump(expected):
        raise ValueError('Unsupported topology source: final step differs')
    function.body.pop()
    function.decorator_list = []
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), '<level4_topology_v2>', 'exec'),
         method.__func__.__globals__, namespace)
    return types.MethodType(namespace[function.name], method.__self__)


def clone(value):
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if hasattr(value, 'clone'):
        return value.clone()
    return copy.deepcopy(value)


class PhysicalTransaction:
    """Restore current MPM state, tool history, pending inputs and clocks.

    Forward-only: MPM adjoint/history replay is unsupported. MPM current and
    previous slots are restored; no training/backpropagation runs are allowed.
    """
    def __init__(self, scene):
        self.scene = scene

    def __enter__(self):
        scene = self.scene
        self.counter = scene.sim.cur_substep_global
        self.local = scene.sim.cur_substep_local
        self.last_local = scene.sim.max_substeps_local - 1
        self.time = scene.t
        self.state = scene.get_state()
        mpm = scene.sim.mpm_solver
        self.previous_index = (self.counter - 1) % scene.sim.max_substeps_local
        self.previous = mpm.get_state(self.previous_index)
        self.tools = []
        for tool in scene.sim.tool_solver.entities:
            fields = {key: getattr(tool, key).to_numpy().copy()
                      for key in ('pos', 'quat', 'vel', 'ang', 'latest_pos', 'latest_quat')}
            self.tools.append((tool, fields))
        self.pending = [(entity, clone(entity._tgt)) for entity in scene.sim.entities
                        if hasattr(entity, '_tgt')]
        return self

    def __exit__(self, exc_type, exc, tb):
        scene = self.scene
        # Call the observed restore so clock capture records branch rollback.
        scene.set_state(self.state)
        scene.sim.mpm_solver.set_state(self.previous_index, self.previous)
        scene.sim.mpm_solver.set_state(self.local, self.state.mpm_solver)
        scene.sim.mpm_solver.set_state(0, self.state.mpm_solver)
        # Counter0 has previous index max_substeps_local-1. Preserve that
        # predecessor too, rather than exposing an unrelated stale ring slot.
        scene.sim.mpm_solver.set_state(self.last_local, self.previous)
        for tool, fields in self.tools:
            for key, value in fields.items():
                if key in ('pos', 'quat', 'vel', 'ang'):
                    restored = value.copy()
                    restored[0] = value[self.local]
                    restored[self.last_local] = value[self.previous_index]
                    value = restored
                getattr(tool, key).from_numpy(value)
            tool.update_mesh_pose(0)
        for entity, pending in self.pending:
            entity._tgt = pending
        scene.t = self.time
        return False


def install(workspace, expected_source_sha256):
    topology = workspace.topology
    if topology is None:
        raise ValueError('Topology diagnostics must remain enabled')
    source = inspect.getsourcefile(type(topology))
    with open(source, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if digest != expected_source_sha256:
        raise ValueError('Unreviewed topology implementation')
    if getattr(topology, '_level4_transaction_installed', False):
        raise ValueError('Adapter already installed')
    check = without_final_step(topology.check)
    def transactional_check():
        with PhysicalTransaction(workspace.scene):
            return check()
    topology.check = transactional_check
    topology._level4_transaction_installed = True
    return {'execution_version': 'level4_forward_topology_transaction_v3',
            'topology_source_sha256': digest, 'legacy_model_alignment': 'unverified',
            'retained_topology_steps': 0, 'forward_only': True}
