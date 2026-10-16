"""Allocation-preserving CUDA checks. Never create a scene, model, or trajectory."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
import zipfile

ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'TI_VISIBLE_DEVICE',
            'CUDA_DEVICE_ORDER', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CONDA_PREFIX',
            'VIRTUAL_ENV', 'PBS_JOBID')
PROBES = ('driver', 'torch', 'taichi', 'generator_import_taichi')


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def selected_environment():
    # Explicit allowlist; no credentials or general environment dump.
    return {key: os.environ.get(key) for key in ENV_KEYS}


def cuda_driver():
    """Query CUDA inside the existing allocation; never change visibility."""
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    code = driver.cuInit(0)
    row = {'cuInit_code': int(code)}
    if code:
        for name in ('cuGetErrorName', 'cuGetErrorString'):
            value = ctypes.c_char_p()
            fn = getattr(driver, name)
            fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            fn.restype = ctypes.c_int
            if fn(code, ctypes.byref(value)) == 0:
                row[name] = value.value.decode() if value.value else None
        return row
    count = ctypes.c_int()
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    count_code = driver.cuDeviceGetCount(ctypes.byref(count))
    row.update(device_count_code=int(count_code), device_count=int(count.value))
    return row


def require_single_device(driver):
    if driver.get('cuInit_code') != 0 or driver.get('device_count_code') != 0:
        raise RuntimeError('CUDA driver initialization/query failed; no allocation performed')
    if driver.get('device_count') != 1:
        raise RuntimeError('Expected one scheduler-visible CUDA device; do not allocate on ambiguous devices')


def loaded_gpu_libraries():
    maps = Path('/proc/self/maps')
    if not maps.exists():
        return []
    return sorted({line.split()[-1] for line in maps.read_text().splitlines()
                   if '/' in line and any(s in line for s in ('libcuda', 'libnvidia', 'libcudart', 'libtaichi'))})


def child(mode, output, runtime, memory_gb):
    before = selected_environment()
    row = {'mode': mode, 'created_epoch_s': time.time(), 'hostname': socket.gethostname(),
           'environment_before': before, 'new_simulations': 0, 'new_Dyn_rollouts': 0,
           'candidate_pool_released': False}
    exit_code = 0
    try:
        if not os.environ.get('PBS_JOBID'):
            raise RuntimeError('This GPU diagnostic must run inside a PBS allocation')
        if mode == 'generator_import_taichi':
            os.chdir(runtime)
            sys.path.insert(0, str(runtime))
            spec = importlib.util.spec_from_file_location('_level4_cuda_import_only', runtime / 'sim/generate.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            row['generation_module_imported_only'] = True
            row['workspace_constructed'] = False
            if selected_environment() != before:
                raise RuntimeError('Generator import changed device binding; scene remains unconstructed')
        driver = cuda_driver()
        row['driver'] = driver
        require_single_device(driver)
        if mode == 'torch':
            import torch
            row.update(torch_version=torch.__version__, torch_compiled_cuda=torch.version.cuda,
                       torch_visible_count=torch.cuda.device_count())
            if row['torch_visible_count'] != 1:
                raise RuntimeError('Torch device enumeration disagrees with the single allocated device')
            # Logical0 is the sole device in the unchanged allocation mask.
            tensor = torch.tensor([2.0], device='cuda:0')
            value = float((tensor + 1).cpu().item())
            if value != 3.0:
                raise RuntimeError('Torch scalar CUDA calculation differs')
            row.update(torch_device=torch.cuda.get_device_name(0), scalar_result=value)
        elif mode in ('taichi', 'generator_import_taichi'):
            import taichi as ti
            ti.init(arch=ti.cuda, device_memory_GB=memory_gb, enable_fallback=False)
            if ti.cfg.arch != ti.cuda:
                raise RuntimeError('Taichi backend is not CUDA')
            value = ti.field(dtype=ti.i32, shape=())
            value[None] = 7
            # Uses a CUDA scalar field transfer, no MPM step or simulator scene.
            if int(value[None]) != 7:
                raise RuntimeError('Taichi CUDA scalar transfer differs')
            row.update(taichi_version=list(ti.__version__), taichi_arch=str(ti.cfg.arch),
                       scalar_result=int(value[None]), device_memory_GB=memory_gb)
            ti.reset()
        row['status'] = 'passed'
    except BaseException as exc:
        exit_code = 1
        row.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        row['environment_after'] = selected_environment()
        row['binding_unchanged'] = row['environment_after'] == before
        row['loaded_gpu_libraries'] = loaded_gpu_libraries()
        row['exit_code'] = exit_code
        write(output, row)
    return exit_code


def run_probes(script, runtime, output, memory_gb, *, runner=subprocess.run):
    """All branches are diagnosed; none modifies scheduler GPU masks."""
    rows = {}
    for mode in PROBES:
        command = [sys.executable, '-B', '-u', str(script), 'child', '--mode', mode,
                   '--runtime', str(runtime), '--output', str(output / (mode + '.json')),
                   '--memory-gb', str(memory_gb)]
        started = time.monotonic()
        with (output / (mode + '.log')).open('xb') as log:
            try:
                result = runner(command, cwd=runtime, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                row = {'exit_code': int(result.returncode), 'timed_out': False}
            except subprocess.TimeoutExpired:
                row = {'exit_code': None, 'timed_out': True, 'status': 'timeout'}
        row.update(command=command, elapsed_s=time.monotonic() - started)
        write(output / (mode + '_exit.json'), row)
        report_path = output / (mode + '.json')
        row['report_present'] = report_path.exists()
        row['passed'] = (row['exit_code'] == 0 and row['report_present'] and
                         json.loads(report_path.read_text(encoding='utf-8'))['status'] == 'passed' and
                         json.loads(report_path.read_text(encoding='utf-8'))['binding_unchanged'] is True)
        rows[mode] = row
    return rows


def collect(request_path, output):
    try:
        from prepare_level4_shared_input import assemble_runtime, digest
        from resume_level4_shared_input import environment, check_files
    except ModuleNotFoundError as exc:
        if exc.name not in ('prepare_level4_shared_input', 'resume_level4_shared_input'):
            raise
        from scripts.prepare_level4_shared_input import assemble_runtime, digest
        from scripts.resume_level4_shared_input import environment, check_files
    output.mkdir(exist_ok=False)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    started = time.monotonic()
    exit_code = 1
    try:
        if (request['kind'] != 'cuda_binding_diagnostic_v1' or request['new_simulations_max'] != 0
                or request['new_Dyn_rollouts'] != 0 or request['candidate_pool_released'] is not False):
            raise ValueError('Only environment diagnostics with zero simulations are allowed')
        if request.get('taichi_memory_gb') != 4.0:
            raise ValueError('Memory reservation must match the failed simulator initialization: 4GiB')
        if not os.environ.get('PBS_JOBID'):
            raise RuntimeError('PBS allocation required before CUDA access')
        actual = environment()
        write(output / 'environment.json', actual)
        write(output / 'gpu_binding_before.json', selected_environment())
        if actual != request['expected_compute_environment']:
            raise ValueError('Compute Python/package environment differs')
        check_files(request['required_files'])
        runtime = assemble_runtime(request, request_path.parent, output.parent)
        from level4.artifacts import runtime_commit
        runtime_commit(runtime)
        import shutil
        shutil.copyfile(runtime.parent / 'manifest.json', output / 'sealed_runtime_manifest.json')
        for filename, command in (
            ('gpu_query.txt', ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,driver_version', '--format=csv']),
            ('device_nodes.txt', ['bash', '-c', 'ls -l /dev/nvidia*']),
        ):
            result = subprocess.run(command, capture_output=True, timeout=20)
            with (output / filename).open('xb') as stream:
                stream.write(result.stdout + result.stderr)
            write(output / (filename + '.exit.json'), {'command': command, 'exit_code': result.returncode})
        rows = run_probes(Path(__file__).resolve(), runtime, output, request['taichi_memory_gb'])
        exit_code = 0 if all(r['passed'] for r in rows.values()) else 1
        report = {'status': 'CUDA_diagnostic_passed_no_pool_release' if exit_code == 0 else 'CUDA_diagnostic_failed',
                  'probes': rows, 'root_cause': 'requires_manual_evidence_review',
                  'binding_policy': 'preserve_inherited_mask;no_environment_override'}
        check_files(request['required_files'])
    except BaseException as exc:
        exit_code = 1
        report = {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()}
        write(output / 'probe_failure.json', report)
    report.update(new_simulations=0, new_Dyn_rollouts=0, candidate_pool_released=False)
    write(output / 'diagnostic_report.json', report)
    write(output / 'request.json', request)
    import shutil
    shutil.copyfile(output.parent / 'toolkit.zip', output / 'submitted_toolkit.zip')
    write(output / 'stage_exit.json', {'exit_code': exit_code, 'elapsed_s': time.monotonic() - started,
                                     'scope': 'CUDA_environment_only', 'new_simulations': 0,
                                     'new_Dyn_rollouts': 0, 'candidate_pool_released': False})
    write(output / 'evidence_manifest.json', {'files_sha256': {
        p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}})
    with zipfile.ZipFile(output.parent / 'diagnostic_evidence.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in output.rglob('*'):
            if p.is_file():
                archive.write(p, p.relative_to(output).as_posix())
    print(json.dumps(report))
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    small = sub.add_parser('child')
    small.add_argument('--mode', choices=PROBES, dest='probe', required=True)
    small.add_argument('--runtime', type=Path, required=True)
    small.add_argument('--output', type=Path, required=True)
    small.add_argument('--memory-gb', type=float, required=True)
    batch = sub.add_parser('collect')
    batch.add_argument('--request', type=Path, required=True)
    batch.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'child':
        sys.exit(child(args.probe, args.output.resolve(), args.runtime.resolve(), args.memory_gb))
    sys.exit(collect(args.request.resolve(), args.output.resolve()))
