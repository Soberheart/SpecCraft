"""Prospective Level4 adapter preserving the single PBS GPU allocation.

The archived builder and its simulator initialization arguments stay intact.
Only the generator's imported get_scene callable is adapted in this process;
the derived function and binding checks are saved alongside the capture.
No environment mapping, historical source file, or solver step is patched.
"""
import ast
import copy
import ctypes
import os
from pathlib import Path
import re
import traceback
import uuid

from .artifacts import sha256, utc_now, write_json

ENV_KEYS = ('PBS_JOBID', 'CUDA_VISIBLE_DEVICES', 'TI_VISIBLE_DEVICE',
            'NVIDIA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')
GPU_UUID = r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'


def binding_environment():
    return {key: os.environ.get(key) for key in ENV_KEYS}


def validate_binding(environment, cuda_id):
    if not environment.get('PBS_JOBID'):
        raise ValueError('PBS allocation required for the prospective binding adapter')
    # Configuration indexes the sole logical device, never the physical GPU ID.
    if isinstance(cuda_id, bool) or not isinstance(cuda_id, int) or cuda_id != 0:
        raise ValueError('Only logical cuda_id=0 inside the inherited allocation is allowed')
    mask = environment.get('CUDA_VISIBLE_DEVICES')
    if not isinstance(mask, str) or not re.fullmatch(r'(?:0|[1-9][0-9]*|' + GPU_UUID + r')', mask):
        raise ValueError('A single numeric or full GPU UUID allocation mask is required')
    if environment.get('TI_VISIBLE_DEVICE') not in (None, '0'):
        raise ValueError('TI_VISIBLE_DEVICE must be absent or logical0; allocation is never rewritten')
    return {'logical_device': 0, 'scheduler_mask': mask, 'mask_preserved': True}


def cuda_driver_identity():
    """Initialize/query only within the inherited mask; record logical0 UUID."""
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    code = int(driver.cuInit(0))
    row = {'cuInit_code': code}
    if code:
        return row
    count = ctypes.c_int()
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    row.update(device_count_code=int(driver.cuDeviceGetCount(ctypes.byref(count))),
               device_count=int(count.value))
    if row['device_count_code'] != 0 or row['device_count'] != 1:
        return row
    device = ctypes.c_int()
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    row['logical0_device_code'] = int(driver.cuDeviceGet(ctypes.byref(device), 0))
    class DeviceUUID(ctypes.Structure):
        _fields_ = [('bytes', ctypes.c_ubyte * 16)]
    identity = DeviceUUID()
    query = driver.cuDeviceGetUuid
    query.argtypes = [ctypes.POINTER(DeviceUUID), ctypes.c_int]
    query.restype = ctypes.c_int
    row['logical0_uuid_code'] = int(query(ctypes.byref(identity), device))
    row['logical0_uuid'] = 'GPU-' + str(uuid.UUID(bytes=bytes(identity.bytes)))
    return row


def validate_driver(row, mask):
    if any(row.get(key) != 0 for key in ('cuInit_code', 'device_count_code',
                                        'logical0_device_code', 'logical0_uuid_code')):
        raise RuntimeError('CUDA driver initialization/device identity query failed')
    if row.get('device_count') != 1:
        raise RuntimeError('Exactly one driver-visible device required before simulator initialization')
    identity = row.get('logical0_uuid')
    if not isinstance(identity, str) or not re.fullmatch(GPU_UUID, identity):
        raise RuntimeError('Valid logical0 CUDA UUID required')
    if mask.startswith('GPU-') and identity.lower() != mask.lower():
        raise RuntimeError('Logical0 UUID differs from the PBS allocation mask')


def require_cuda_backend():
    import taichi as ti
    if ti.cfg.arch != ti.cuda:
        raise RuntimeError('Simulator Taichi backend is not CUDA; scene creation rejected')
    return str(ti.cfg.arch)


def derive_function(source):
    """Remove exactly the two known overrides; guard the original init call."""
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_scene']
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError('Expected one undecorated archived get_scene function')
    original = matches[0]
    expected = ast.parse("os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_id)\n"
                         "os.environ['TI_VISIBLE_DEVICE'] = str(config.cuda_id)\n"
                         "us.init(seed=config.sim.seed, allocate_gpu_memory=config.sim.allocate_gpu_memory, "
                         "precision=str(config.sim.precision), logging_level=config.sim.logging_level)").body
    if len(original.body) < 3 or any(ast.dump(a) != ast.dump(b) for a, b in zip(original.body[:3], expected)):
        raise ValueError('Archived binding/init statements differ; explicit review required')
    adapted = copy.deepcopy(original)
    adapted.body = adapted.body[2:]
    adapted.body[0].value = ast.Call(func=ast.Name(id='__level4_guarded_initialize', ctx=ast.Load()),
        args=[ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                         body=copy.deepcopy(original.body[2].value)),
              ast.Name(id='config', ctx=ast.Load())], keywords=[])
    # The complete scene/control construction after init must remain identical.
    assert [ast.dump(n) for n in adapted.body[1:]] == [ast.dump(n) for n in original.body[3:]]
    module = ast.fix_missing_locations(ast.Module(body=[adapted], type_ignores=[]))
    return ast.unparse(module) + '\n'


def install(generator, builder_path, expected_sha256, output):
    """Install only at an explicitly requested capture, before workspace creation."""
    builder_path, output = Path(builder_path).resolve(), Path(output)
    if sha256(builder_path) != expected_sha256:
        raise ValueError('Archived GPU builder source changed')
    original = generator.get_scene
    if Path(original.__code__.co_filename).resolve() != builder_path:
        raise ValueError('Generator get_scene does not originate from the reviewed builder')
    source = builder_path.read_text(encoding='utf-8')
    derived = derive_function(source)
    # Verify that the imported callable also matches the archived function.
    original_tree = ast.parse(source)
    original_node = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_scene')
    compiled = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original_node], type_ignores=[])),
                 str(builder_path), 'exec'), compiled)
    fields = ('co_code', 'co_names', 'co_consts', 'co_varnames')
    if any(getattr(original.__code__, key) != getattr(compiled['get_scene'].__code__, key) for key in fields):
        raise ValueError('Loaded builder callable differs from the reviewed source')
    derived_path = output / 'gpu_bound_get_scene.py'
    with derived_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(derived)
    def guarded_initialize(initialize, config):
        before = binding_environment()
        row = {'created_utc': utc_now(), 'environment_before': before,
               'config_cuda_id': config.cuda_id, 'original_builder_sha256': expected_sha256,
               'derived_function_sha256': sha256(derived_path), 'candidate_pool_released': False}
        write_json(output / 'gpu_binding_before_init.json', row)
        try:
            row['binding'] = validate_binding(before, config.cuda_id)
            row['driver'] = cuda_driver_identity()
            validate_driver(row['driver'], before['CUDA_VISIBLE_DEVICES'])
            if binding_environment() != before:
                raise RuntimeError('Allocation binding changed during driver preflight')
            write_json(output / 'gpu_binding_driver.json', row['driver'])
            initialize()  # Original arguments and simulator init, no extra MPM step.
            row['taichi_arch'] = require_cuda_backend()
            if binding_environment() != before:
                raise RuntimeError('Allocation binding changed during simulator initialization')
            row['status'] = 'passed_simulator_initialization_only'
        except BaseException as exc:
            row.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            row['environment_after'] = binding_environment()
            row['binding_unchanged'] = row['environment_after'] == before
            write_json(output / 'gpu_binding_after_init.json', row)
    namespace = dict(original.__globals__)
    namespace['__level4_guarded_initialize'] = guarded_initialize
    exec(compile(derived, str(derived_path), 'exec'), namespace)
    generator.get_scene = namespace['get_scene']
    receipt = {'created_utc': utc_now(), 'version': 'level4_PBS_binding_v1',
               'original_builder_sha256': expected_sha256, 'derived_function_sha256': sha256(derived_path),
               'overrides_removed': ['CUDA_VISIBLE_DEVICES', 'TI_VISIBLE_DEVICE'],
               'original_initialization_arguments_preserved': True,
               'post_init_scene_construction_AST_unchanged': True,
               'original_builder_file_unchanged': sha256(builder_path) == expected_sha256,
               'scope': 'prospective_capture_callable_only;no_environment_monkeypatch',
               'GPU_verification': 'pending_workspace_initialization', 'candidate_pool_released': False}
    write_json(output / 'gpu_binding_adapter.json', receipt)
    return receipt
