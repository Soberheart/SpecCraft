"""Instrument two independent nominal runs without changing physics or reset gates."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.array(value, copy=True)
    if result.dtype.kind not in 'biuf':
        raise ValueError('State must contain numeric arrays, not objects')
    return result


class Recorder:
    """Read state at existing transitions; never advance or replace simulator state."""
    def __init__(self, output):
        self.output = Path(output)
        self.rows = []

    def save(self, workspace, name):
        state = workspace.scene.get_state()
        packet = {}
        for key in ('pos', 'vel', 'C', 'F', 'active', 'mat_id'):
            value = getattr(state.mpm_solver, key)
            if value is None:
                raise ValueError('Missing complete MPM state: ' + key)
            packet['mpm_' + key] = array(value)
        for i, tool in enumerate(state.tool_solver.data):
            for key in ('pos', 'quat', 'vel', 'ang'):
                packet['tool_%03d_%s' % (i, key)] = array(getattr(tool, key))
        path = self.output / (name + '.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **packet)
        self.rows.append({'stage': name, 'file': path.name, 'sha256': digest(path),
                          's_global': int(state.s_global), 'scene_t': int(workspace.scene.t),
                          'fields': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in packet.items()}})

    def attach(self, workspace):
        _attach_legacy(self, workspace)


def install_shared_reset(workspace, packet, recorder):
    # Import the delivered helper, rather than the old archived level4 package.
    helper_path = Path(__file__).with_name('shared_reset.py')
    if helper_path.is_file():
        spec = importlib.util.spec_from_file_location('_level4_shared_reset', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        loader = helper.load_shared_state
    else:
        from level4.shared_reset import load_shared_state
        loader = load_shared_state

    original_reset = workspace.reset

    def reset():
        # The original reset still creates the scene directory/config and
        # resets the controller/topology. A populated init_state skips warmup.
        state = workspace.scene.get_state()
        loader(packet, state)
        workspace.init_state = state
        result = original_reset()
        recorder.save(workspace, 'reset_returned')
        return result
    workspace.reset = reset

def attach_recorder(recorder, workspace, shared_state=None):
    """Keep the legacy path intact; shared restore is a separate explicit mode."""
    if shared_state is not None:
        install_shared_reset(workspace, shared_state, recorder)
        recorder.save(workspace, 'constructed')
        original_log=workspace.log
        def log(step,*args,**kwargs):
            if step==0:
                recorder.save(workspace,'logged_frame0')
            return original_log(step,*args,**kwargs)
        workspace.log=log
        return
    recorder.attach(workspace)


def _attach_legacy(self, workspace):
        self.save(workspace, 'constructed')
        original_step = workspace.scene.step
        original_reset = workspace.scene.reset
        original_log = workspace.log
        original_workspace_reset = workspace.reset
        warmup = int(workspace.cfg.warmup_horizon)
        counter = {'steps': 0, 'in_reset': False, 'scene_resets': 0}

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            counter['steps'] += 1
            if counter['in_reset'] and counter['steps'] in {1, max(1, warmup // 2), warmup}:
                self.save(workspace, 'warmup_%04d' % counter['steps'])
            return result

        def scene_reset(*args, **kwargs):
            counter['scene_resets'] += 1
            label = 'scene_reset_%02d' % counter['scene_resets']
            self.save(workspace, label + '_before')
            result = original_reset(*args, **kwargs)
            self.save(workspace, label + '_after')
            return result

        def reset(*args, **kwargs):
            counter['in_reset'] = True
            try:
                result = original_workspace_reset(*args, **kwargs)
            finally:
                counter['in_reset'] = False
            self.save(workspace, 'reset_returned')
            return result

        def log(step, *args, **kwargs):
            if step == 0:
                self.save(workspace, 'logged_frame0')
            return original_log(step, *args, **kwargs)

        workspace.scene.step = step
        workspace.scene.reset = scene_reset
        workspace.reset = reset
        workspace.log = log


def compare_states(first, second, unit_to_meter=.2):
    first, second = Path(first), Path(second)
    records = []
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('State field sets differ')
        for key in a.files:
            x, y = a[key].astype(float), b[key].astype(float)
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('State field shape/nonfinite mismatch: ' + key)
            error = y - x
            factor = 1000 * unit_to_meter if key == 'mpm_pos' else 1.
            records.append({'field': key, 'max_abs': float(np.max(np.abs(error))) * factor,
                            'rms': float(np.sqrt(np.mean(error**2))) * factor,
                            'unit': 'mm' if key == 'mpm_pos' else 'native_simulator_state_unit',
                            'exactly_equal': bool(np.array_equal(a[key], b[key]))})
    return records


def capture(runtime, config, output, shared_state=None):
    output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(output)
    try:
        os.chdir(runtime)
        sys.path.insert(0, str(runtime))
        from omegaconf import OmegaConf
        module_spec = importlib.util.spec_from_file_location('_level4_initialization_generator', runtime / 'sim/generate.py')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        if module.DEBUG:
            raise ValueError('Debugger suppresses warmup; diagnostic requires normal execution')
        cfg = OmegaConf.load(config)
        cfg.log.base_dir = str(output / 'scenes')
        (output / 'scenes').mkdir()
        OmegaConf.save(cfg, output / 'requested_config.yaml')
        workspace = module.GenerationWorkspace(cfg)
        static = workspace.scene.sim.mpm_solver.particles_info.to_numpy()
        with (output/'static_material_info.npz').open('xb') as out:
            np.savez_compressed(out, **{key: array(static[key]) for key in ('mat_idx','mu','lam','mass')})
        attach_recorder(recorder,workspace,shared_state)
        workspace.run()
        write_json(output / 'capture_report.json', {'kind': 'initialization_diagnostic_not_reset_acceptance',
                   'created_utc': utc_now(), 'status': 'captured', 'states': recorder.rows,
                   'reset_gate_changed': False, 'primary_frame': 60, 'primary_s': 1.920,
                   'shared_reset_state': str(shared_state) if shared_state else None,
                   'warmup_replayed': shared_state is None,
                   'mat_id_note': 'mpm_mat_id is unused zero placeholder; static_material_info saves actual map',
                   'time_note': 'Legacy step0 log is after one scene.step; no relabeling in this diagnostic'})
    except Exception as exc:
        write_json(output / 'capture_failure.json', {'status': 'failed', 'error': repr(exc), 'states': recorder.rows,
                                                    'traceback': traceback.format_exc()})
        raise


def _collect(request_path, output):
    request = json.loads(request_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    if digest(Path(__file__)) != request['probe_source_sha256']:
        raise ValueError('Frozen diagnostic collector source changed')
    with Path(__file__).open('rb') as inp, (output / 'collector.py').open('xb') as out:
        shutil.copyfileobj(inp, out)
    if request.get('shared_reset_state'):
        helper=Path(__file__).with_name('shared_reset.py')
        if digest(helper)!=request['shared_reset_helper_sha256']:
            raise ValueError('Frozen shared-state helper changed')
        with helper.open('rb') as inp,(output/'shared_reset.py').open('xb') as out:
            shutil.copyfileobj(inp,out)
    actual = {'python': sys.version, 'packages': dict(sorted((d.metadata['Name'].lower(), d.version)
              for d in importlib.metadata.distributions() if d.metadata.get('Name')))}
    write_json(output / 'environment.json', actual)
    if actual != request['expected_compute_environment']:
        raise ValueError('Compute environment differs from the reviewed actual environment; no simulation started')
    runtime = Path(request['runtime'])
    for name, record in request['required_files'].items():
        path = Path(name)
        if path.is_symlink() or path.stat().st_size != record['size'] or digest(path) != record['sha256']:
            raise ValueError('Archived input/source changed: ' + name)
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise ValueError('At least 2 GiB free required for diagnostic evidence')
    if request.get('shared_reset_state'):
        # Keep the source body in the evidence, not just its remote path/hash.
        with Path(request['shared_reset_state']).open('rb') as inp,(output/'shared_source_state.npz').open('xb') as out:
            shutil.copyfileobj(inp,out)
        if digest(output/'shared_source_state.npz')!=request['required_files'][request['shared_reset_state']]['sha256']:
            raise ValueError('Shared source changed during evidence copy; no simulation started')
    results = []
    for attempt in (1, 2):
        target = output / ('attempt_%d' % attempt)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'capture', '--runtime', str(runtime),
                   '--config', request['config'], '--output', str(target)]
        if request.get('shared_reset_state'):
            command+=['--shared-state',str(output/'shared_source_state.npz')]
        started = time.monotonic()
        with (output / ('attempt_%d.log' % attempt)).open('x') as log:
            process = subprocess.run(command, cwd=runtime, stdout=log, stderr=subprocess.STDOUT)
        result = {'attempt': attempt, 'exit_code': process.returncode, 'elapsed_s': time.monotonic() - started,
                  'command': command}
        results.append(result)
        write_json(output / ('attempt_%d_exit.json' % attempt), result)
    comparison = {}
    shared_audit = None
    if all(r['exit_code'] == 0 for r in results):
        for row in json.loads((output / 'attempt_1/capture_report.json').read_text(encoding='utf-8'))['states']:
            name = row['file']
            comparison[row['stage']] = compare_states(output / 'attempt_1' / name, output / 'attempt_2' / name)
        initial_errors = []
        if request.get('shared_reset_state'):
            for attempt in (1,2):
                initial_errors.append({'attempt':attempt,'relation':'shared_saved_pre_action_state',
                    'fields':compare_states(output/'shared_source_state.npz',output/('attempt_%d/reset_returned.npz'%attempt))})
            with np.load(output/'attempt_1/static_material_info.npz',allow_pickle=False) as a,np.load(output/'attempt_2/static_material_info.npz',allow_pickle=False) as b:
                material_equal=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
            exact=all(row['exactly_equal'] for entry in initial_errors for row in entry['fields'])
            with np.load(output/'attempt_1/logged_frame0.npz',allow_pickle=False) as a,np.load(output/'attempt_2/logged_frame0.npz',allow_pickle=False) as b:
                frame0_mm=float(np.max(np.abs(a['mpm_pos'].astype(float)-b['mpm_pos'].astype(float)))*200)
            shared_audit={'kind':'shared_state_initialization_audit_not_pool_gate',
                          'same_saved_state_restored_exactly':exact,'actual_material_map_equal':material_equal,
                          'frame0_pair_max_abs_mm':frame0_mm,'unchanged_limit_mm':.001,
                          'frame0_within_unchanged_limit':frame0_mm<=.001,
                          'status':'passed' if exact and material_equal and frame0_mm<=.001 else 'failed',
                          'new_initial_state_source':request['shared_reset_state'],
                          'shared_source_sha256':digest(output/'shared_source_state.npz'),
                          'historical_actual_material_map_not_captured':True,
                          'historical_input_not_relabelled':True,'candidate_pool_released':False}
        else:
            with np.load(request['reference_npz'], allow_pickle=False) as packet:
                reference = packet['planner_initial_positions'].astype(float)
        for attempt in (1, 2):
            if request.get('shared_reset_state'):
                continue
            with np.load(output / ('attempt_%d/logged_frame0.npz' % attempt), allow_pickle=False) as packet:
                positions = packet['mpm_pos'][packet['mpm_active'].astype(bool)].astype(float)
            if positions.shape != reference.shape:
                initial_errors.append({'attempt': attempt, 'status': 'unknown', 'reason': 'active_count_differs'})
            else:
                error_mm = float(np.max(np.abs(positions - reference)) * 200)
                initial_errors.append({'attempt': attempt, 'max_abs_error_mm': error_mm,
                                       'within_unchanged_position_limit': error_mm <= .001})
    else:
        initial_errors = []
    report = {'kind': 'two_process_initialization_diagnostic_not_reset_gate', 'created_utc': utc_now(),
              'status': 'captured' if comparison else 'capture_failed', 'attempts': results,
              'stage_comparison': comparison,
              'provisional_position_limit_mm': .001, 'reset_gate_changed': False,
              'reset_gate_passed': False, 'reason': 'Diagnostic only; no acceptance or candidate-pool release',
              'new_model_prediction': False, 'nominal_simulations': 2}
    report['shared_source_state_comparison' if request.get('shared_reset_state') else
           'historical_reference_position_comparison']=initial_errors
    if shared_audit is not None:
        report['shared_initialization_audit']=shared_audit
        write_json(output/'shared_initialization_audit.json',shared_audit)
    write_json(output / 'diagnostic_report.json', report)
    bundle_evidence(output)
    if not comparison:
        raise RuntimeError('Capture failed; diagnostic bundle retained')
    if shared_audit is not None and shared_audit['status']!='passed':
        raise RuntimeError('Shared-state audit failed; diagnostic bundle retained; pool remains blocked')


def bundle_evidence(output):
    files = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'evidence_manifest.json', {'kind': 'initialization_diagnostic_evidence',
                                                'generated_utc': utc_now(), 'files_sha256': files})
    import zipfile
    archive = output.parent / 'diagnostic_evidence.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob('*'):
            if path.is_file():
                z.write(path, path.relative_to(output).as_posix())
    print(json.dumps({'status': 'evidence_saved', 'archive': str(archive), 'archive_sha256': digest(archive),
                      'archive_bytes': archive.stat().st_size, 'report': str(output / 'diagnostic_report.json')}))


def collect(request_path, output):
    if output.exists():
        raise FileExistsError('Diagnostic output exists; retain earlier attempt')
    try:
        _collect(request_path, output)
    except Exception as exc:
        # Include even failed preflight/analysis evidence in a downloadable
        # package. A failed scientific result never becomes a successful run.
        if output.exists() and not (output.parent/'diagnostic_evidence.zip').exists():
            write_json(output/'collector_failure.json',{'created_utc':utc_now(),'error':repr(exc),
                       'traceback':traceback.format_exc(),'candidate_pool_released':False})
            bundle_evidence(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    child = sub.add_parser('capture')
    child.add_argument('--runtime', type=Path, required=True)
    child.add_argument('--config', type=Path, required=True)
    child.add_argument('--output', type=Path, required=True)
    child.add_argument('--shared-state',type=Path)
    outer = sub.add_parser('collect')
    outer.add_argument('--request', type=Path, required=True)
    outer.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        capture(args.runtime.resolve(), args.config.resolve(), args.output.resolve(),
                args.shared_state.resolve() if args.shared_state else None)
    else:
        collect(args.request.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
