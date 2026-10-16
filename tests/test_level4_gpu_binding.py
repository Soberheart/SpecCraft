"""CPU regression for the real archived builder route; CUDA is mocked."""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import uuid
from unittest.mock import Mock, patch

from level4.artifacts import read_json, sha256
from level4 import gpu_binding as guard

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / 'sim/generate/builder.py'
IDENTITY = 'GPU-88d8b15f-cd4f-6169-6056-9e8d12901aac'


def environment(mask=IDENTITY, **kwargs):
    return dict(PBS_JOBID='123.pbs101', CUDA_VISIBLE_DEVICES=mask,
                TI_VISIBLE_DEVICE=None, NVIDIA_VISIBLE_DEVICES=None,
                CUDA_DEVICE_ORDER='PCI_BUS_ID', **kwargs)


def driver(identity=IDENTITY, count=1, code=0):
    return dict(cuInit_code=code, device_count_code=0, device_count=count,
                logical0_device_code=0, logical0_uuid_code=0, logical0_uuid=identity)


def fixture():
    scene = Mock()
    simulator = NS(init=Mock(), Scene=Mock(return_value=scene),
                   options=NS(**{k: Mock() for k in ('SimOptions', 'ToolOptions', 'MPMOptions', 'ViewerOptions')}))
    namespace = {'os': os, 'us': simulator, 'get_entity': Mock(), 'get_ee': Mock(return_value='planner')}
    tree = ast.parse(BUILDER.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_scene')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(BUILDER), 'exec'), namespace)
    generator = NS(get_scene=namespace['get_scene'])
    sim = NS(seed=0, allocate_gpu_memory=4, precision=32, logging_level='info',
             max_substeps_local=50, gravity=[0, 0, -9.8], step_dt=.002, substep_dt=.0002,
             tool=NS(floor_height=0), mpm=NS(particle_diameter=.01, grid_density=64,
                                          lower_bound=[-1]*3, upper_bound=[1]*3))
    config = NS(cuda_id=0, sim=sim, render=False, entities={}, ee=NS())
    return generator, simulator, scene, config


class GPUBindingTests(unittest.TestCase):
    def test_single_UUID_and_nonzero_physical_numeric_masks_preserved(self):
        for mask in (IDENTITY, '3'):
            with self.subTest(mask=mask), tempfile.TemporaryDirectory() as d:
                root = Path(d); gen, sim, scene, cfg = fixture(); source_hash = sha256(BUILDER)
                original = gen.get_scene
                guard.install(gen, BUILDER, source_hash, root)
                env = {k: v for k, v in environment(mask).items() if v is not None}
                with patch.dict(os.environ, env, clear=True), \
                        patch.object(guard, 'cuda_driver_identity', return_value=driver()) as probe, \
                        patch.object(guard, 'require_cuda_backend', return_value='Arch.cuda'):
                    result = gen.get_scene(cfg)
                    self.assertEqual(guard.binding_environment(), environment(mask))
                probe.assert_called_once()
                sim.init.assert_called_once_with(seed=0, allocate_gpu_memory=4, precision='32', logging_level='info')
                sim.Scene.assert_called_once(); scene.build.assert_called_once(); scene.reset.assert_called_once()
                self.assertEqual(result, (scene, None, [], 'planner'))
                self.assertIsNot(original, gen.get_scene)
                self.assertEqual(sha256(BUILDER), source_hash)
                row = read_json(root / 'gpu_binding_after_init.json')
                self.assertTrue(row['binding_unchanged'])
                self.assertEqual(row['status'], 'passed_simulator_initialization_only')

    def test_original_unadapted_route_still_uses_legacy_behavior(self):
        gen, sim, scene, cfg = fixture()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': IDENTITY}, clear=True):
            gen.get_scene(cfg)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(os.environ['TI_VISIBLE_DEVICE'], '0')

    def test_missing_ambiguous_or_malformed_bindings_rejected_before_driver(self):
        bad = [None, '', '-1', '0,1', 'all', 'GPU-short', ' 3', '3 ', 'MIG-unknown']
        for mask in bad:
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError): guard.validate_binding(environment(mask), 0)
        for config_id in (1, -1, True, '0', 0.0):
            with self.subTest(config_id=config_id):
                with self.assertRaises(ValueError): guard.validate_binding(environment(), config_id)
        for key, value in (('PBS_JOBID', None), ('TI_VISIBLE_DEVICE', '3')):
            env = environment(); env[key] = value
            with self.assertRaises(ValueError): guard.validate_binding(env, 0)
        env = environment(); env['TI_VISIBLE_DEVICE'] = '0'
        self.assertEqual(guard.validate_binding(env, 0)['logical_device'], 0)

    def test_driver_count_error_or_UUID_mismatch_rejected(self):
        for row in (driver(count=2), driver(code=3), driver('GPU-00000000-0000-0000-0000-000000000000')):
            with self.subTest(row=row), self.assertRaises(RuntimeError): guard.validate_driver(row, IDENTITY)

    def test_ctypes_identity_query_decodes_UUID_and_stops_ambiguous_count(self):
        for count in (1, 2):
            library=Mock()
            library.cuInit.return_value=0
            def get_count(pointer): pointer._obj.value=count; return 0
            def get_device(pointer,index): self.assertEqual(index,0); pointer._obj.value=7; return 0
            def get_uuid(pointer,device):
                self.assertEqual(device.value,7)
                pointer._obj.bytes[:]=uuid.UUID(IDENTITY[4:]).bytes
                return 0
            library.cuDeviceGetCount.side_effect=get_count
            library.cuDeviceGet.side_effect=get_device
            library.cuDeviceGetUuid.side_effect=get_uuid
            with self.subTest(count=count), patch.object(guard.ctypes,'CDLL',return_value=library):
                row=guard.cuda_driver_identity()
            self.assertEqual(row['device_count'],count)
            if count==1:
                self.assertEqual(row['logical0_uuid'],IDENTITY)
                guard.validate_driver(row,IDENTITY)
            else:
                library.cuDeviceGet.assert_not_called();library.cuDeviceGetUuid.assert_not_called()

    def test_failed_driver_does_not_initialize_or_construct_scene_and_saves_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); gen, sim, scene, cfg = fixture()
            guard.install(gen, BUILDER, sha256(BUILDER), root)
            with patch.dict(os.environ, {'PBS_JOBID':'123.pbs101','CUDA_VISIBLE_DEVICES':IDENTITY}, clear=True), \
                    patch.object(guard, 'cuda_driver_identity', return_value=driver(code=3)):
                with self.assertRaises(RuntimeError): gen.get_scene(cfg)
            sim.init.assert_not_called(); sim.Scene.assert_not_called()
            row = read_json(root / 'gpu_binding_after_init.json')
            self.assertEqual(row['status'], 'failed'); self.assertTrue(row['binding_unchanged'])

    def test_missing_PBS_saves_failure_before_any_CUDA_call(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); gen, sim, scene, cfg = fixture()
            guard.install(gen, BUILDER, sha256(BUILDER), root)
            with patch.dict(os.environ, {}, clear=True), patch.object(guard, 'cuda_driver_identity') as probe:
                with self.assertRaises(ValueError): gen.get_scene(cfg)
            probe.assert_not_called(); sim.init.assert_not_called()
            self.assertEqual(read_json(root / 'gpu_binding_after_init.json')['status'], 'failed')

    def test_init_exception_keeps_original_error_and_binding_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); gen, sim, scene, cfg = fixture()
            sim.init.side_effect = RuntimeError('original init error')
            guard.install(gen, BUILDER, sha256(BUILDER), root)
            with patch.dict(os.environ, {'PBS_JOBID':'123.pbs101','CUDA_VISIBLE_DEVICES':IDENTITY}, clear=True), \
                    patch.object(guard, 'cuda_driver_identity', return_value=driver()):
                with self.assertRaisesRegex(RuntimeError, 'original init error'): gen.get_scene(cfg)
            sim.Scene.assert_not_called()
            self.assertIn('original init error', read_json(root / 'gpu_binding_after_init.json')['error'])

    def test_binding_mutation_and_CPU_fallback_fail_before_scene(self):
        for mutation in (True, False):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d); gen, sim, scene, cfg = fixture()
                if mutation: sim.init.side_effect = lambda **kw: os.environ.__setitem__('CUDA_VISIBLE_DEVICES', '0')
                guard.install(gen, BUILDER, sha256(BUILDER), root)
                with patch.dict(os.environ, {'PBS_JOBID':'123.pbs101','CUDA_VISIBLE_DEVICES':IDENTITY}, clear=True), \
                        patch.object(guard, 'cuda_driver_identity', return_value=driver()), \
                        patch.object(guard, 'require_cuda_backend',
                                     side_effect=None if mutation else RuntimeError('CPU fallback'), return_value='Arch.cuda'):
                    with self.assertRaises(RuntimeError): gen.get_scene(cfg)
                sim.Scene.assert_not_called()
                self.assertEqual(read_json(root / 'gpu_binding_after_init.json')['binding_unchanged'], not mutation)

    def test_unknown_source_or_origin_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            gen, sim, scene, cfg = fixture()
            with self.assertRaisesRegex(ValueError, 'source changed'): guard.install(gen, BUILDER, 'a'*64, Path(d))
            gen.get_scene = lambda config: None
            with self.assertRaisesRegex(ValueError, 'originate'): guard.install(gen, BUILDER, sha256(BUILDER), Path(d))
        source = BUILDER.read_text(encoding='utf-8').replace("str(config.cuda_id)", "'0'", 1)
        with self.assertRaisesRegex(ValueError, 'explicit review'): guard.derive_function(source)

    def test_modified_loaded_init_arguments_rejected(self):
        gen, sim, scene, cfg = fixture()
        modified = BUILDER.read_text(encoding='utf-8').replace('config.sim.seed', '42', 1)
        node = next(n for n in ast.parse(modified).body if isinstance(n, ast.FunctionDef) and n.name=='get_scene')
        namespace = dict(gen.get_scene.__globals__)
        exec(compile(ast.Module(body=[node],type_ignores=[]), str(BUILDER), 'exec'), namespace)
        gen.get_scene = namespace['get_scene']
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(ValueError, 'callable differs'):
            guard.install(gen, BUILDER, sha256(BUILDER), Path(d))


if __name__ == '__main__': unittest.main()
