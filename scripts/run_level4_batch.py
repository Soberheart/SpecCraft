"""L4-01 Phase B batch execution runner (one PBS job, one GPU, sequential runs).

Generalises the job-25837718 runner (probe_level4_transaction.py) from two fixed
candidates to a frozen list of runs.  Every run uses the *same* execution path:
the archived runtime plus the frozen job-22 Level 4 overlays, a separate child
process ``probe_level4_candidate_interface.py capture`` with the v3 topology
transaction and the PBS GPU binding guard, and the shared reset state.

Per run it records exit status, exact/near reset equality, GPU binding receipt,
the clock replay, and the tool-interface diagnosis against the v3 nominal
(job 25837718 candidate_000) judged under tool_interface_contract_v2.  Scene
tool offsets (S-runs) enter only the comparison parameters, never prediction.

Stop rule (task sheet L4-01 section 4): a GPU-binding failure stops the job at
once; otherwise the job stops when failed runs (non-zero exit or reset gate
failure) exceed 20% of the planned runs.  Remaining runs stay ``not_executed``.
No Dyn, AE, Renderer, task evaluation or pool release happens here.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

try:
    from prepare_level4_shared_input import assemble_runtime, digest, write
    from resume_level4_shared_input import environment, check_files
except ModuleNotFoundError:
    from scripts.prepare_level4_shared_input import assemble_runtime, digest, write
    from scripts.resume_level4_shared_input import environment, check_files

KIND = 'level4_batch_execution_v1'
RESET_LIMIT_MM = 0.001
SIM_TO_MM = 200.0
MAX_RUNS = 45


def validate_request(request):
    runs = request.get('runs')
    if request.get('kind') != KIND or not isinstance(runs, list) or not runs:
        raise ValueError('Not a Level 4 batch execution request')
    ids = [r['run_id'] for r in runs]
    if len(set(ids)) != len(ids) or len(runs) > MAX_RUNS:
        raise ValueError('Run ids must be unique and at most %d per job' % MAX_RUNS)
    if request['new_simulations_max'] != len(runs) or request['new_Dyn_rollouts'] != 0:
        raise ValueError('Batch execution scope differs from the frozen run list')
    if request.get('candidate_pool_released') is not False:
        raise ValueError('Pool release is never part of execution')
    if request.get('stop_rule') != {'max_failure_fraction': 0.2, 'stop_on_binding_failure': True}:
        raise ValueError('Stop rule differs from task sheet L4-01')
    if request.get('tool_interface_contract') != 'tool_interface_contract_v2':
        raise ValueError('Tool comparisons must use the pre-registered v2 contract')
    if request.get('gpu_binding_policy') != 'preserve_single_PBS_allocation_v1':
        raise ValueError('Batch execution requires the verified PBS binding guard')
    for r in runs:
        offset, action = r['tool_offset_mm'], r['action_mm_mm_deg']
        if len(offset) != 2 or len(action) != 3 or not np.isfinite(offset + action).all():
            raise ValueError('Invalid action/offset for ' + r['run_id'])
        if Path(r['scene']).name != r['scene']:
            raise ValueError('Scene must be a toolkit file name')


def comparison_parameters(run):
    """Executed pose relative to the v3 nominal: action plus whole-tool offset."""
    dx, dy, yaw = (float(v) for v in run['action_mm_mm_deg'])
    ox, oy = (float(v) for v in run['tool_offset_mm'])
    return [dx + ox, dy + oy, yaw]


def reset_check(shared_state, returned):
    with np.load(shared_state, allow_pickle=False) as a, np.load(returned, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return {'status': 'failed', 'reason': 'array set differs'}
        exact = all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
                    for k in a.files)
        if a['mpm_pos'].shape != b['mpm_pos'].shape:
            return {'status': 'failed', 'reason': 'material shape differs'}
        error = float(np.max(np.abs(a['mpm_pos'].astype(float) - b['mpm_pos'].astype(float)))) * SIM_TO_MM
    return {'status': 'passed' if error <= RESET_LIMIT_MM else 'failed', 'exact': bool(exact),
            'max_material_error_mm': error, 'limit_mm': RESET_LIMIT_MM}


def binding_check(target):
    path = target / 'gpu_binding_after_init.json'
    if not path.is_file():
        return {'status': 'failed', 'reason': 'binding receipt missing'}
    receipt = json.loads(path.read_text(encoding='utf-8'))
    ok = receipt.get('status') == 'passed_simulator_initialization_only' and receipt.get('binding_unchanged') is True
    return {'status': 'passed' if ok else 'failed', 'receipt_status': receipt.get('status')}


def verify_reference(reference):
    """The v3 nominal reference must be the hash-verified job-25837718 evidence."""
    manifest = json.loads((reference / 'evidence_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    needed = [n for n in manifest if n.startswith('execution/candidate_000/actual_tools/')
              or n == 'execution/candidate_000/scenes/0000/log.pkl']
    if len(needed) != 64:
        raise ValueError('Nominal reference does not hold 63 tool frames and one log')
    for name in needed:
        if digest(reference / name) != manifest[name]:
            raise ValueError('Nominal reference changed: ' + name)
    return reference / 'execution/candidate_000/scenes/0000/log.pkl', reference / 'execution/candidate_000/actual_tools'


def run(request, tools, output):
    validate_request(request)
    write(output / 'environment.json', environment())
    if environment() != request['expected_compute_environment']:
        raise ValueError('Compute environment changed')
    check_files(request['required_files'])
    need = (2 + 0.1 * len(request['runs'])) * 1024 ** 3
    if shutil.disk_usage(output).free < need:
        raise ValueError('Not enough free disk for the planned runs')
    toolkit = json.loads((tools / 'tool_manifest.json').read_text(encoding='utf-8'))['files_sha256']
    for name, h in toolkit.items():
        if Path(name).name != name or digest(tools / name) != h:
            raise ValueError('Delivered toolkit file changed: ' + name)
    runtime = assemble_runtime(request, tools, output.parent)
    from level4.artifacts import runtime_commit, checked_pool, write_json, utc_now
    from level4.clock_review import review_clock_schedule
    from level4.candidate_interface import diagnose_tool_schedule
    from level4.tool_interface_contract import evaluate
    runtime_commit(runtime)
    pool = Path(request['source_pool'])
    checked_pool(pool)
    if digest(pool / 'pool.json') != request['source_pool_sha256']:
        raise ValueError('Frozen pool differs')
    if digest(runtime / 'sim/generate/builder.py') != request['gpu_builder_source_sha256']:
        raise ValueError('Reviewed GPU builder changed')
    reference_log, reference_tools = verify_reference(Path(request['nominal_reference']['evidence_dir']))
    provenance = output / 'runtime_provenance'
    provenance.mkdir()
    shutil.copyfile(runtime.parent / 'manifest.json', provenance / 'sealed_runtime_manifest.json')
    shutil.copyfile(pool / 'pool.json', output / 'source_pool_manifest.json')
    for name in ('shared_source_state.npz', 'static_material_info.npz', 'source_scene.yaml'):
        shutil.copyfile(pool / name, output / name)
    from omegaconf import OmegaConf
    center = np.asarray(OmegaConf.load(pool / 'source_scene.yaml').actions.grasp.to_pos, dtype=float)
    runs = request['runs']
    ledger = {r['run_id']: {'execution': 'not_executed', 'checks': 'not_run', 'prediction': 'not_run_here'} for r in runs}
    events = output / 'status_events'
    events.mkdir()
    def snapshot():
        write_json(events / ('status_%03d.json' % len(list(events.iterdir()))), ledger)
    snapshot()
    failures, stopped, results = 0, None, {}
    for row in runs:
        rid = row['run_id']
        if stopped:
            continue
        scene = tools / row['scene']
        if digest(scene) != row['scene_sha256']:
            raise ValueError('Frozen scene changed: ' + row['scene'])
        target = output / 'execution' / rid
        shutil.copyfile(scene, output / (rid + '_scene.yaml'))
        command = [sys.executable, '-B', '-u', str(tools / 'probe_level4_candidate_interface.py'), 'capture',
                   '--runtime', str(runtime), '--config', str(scene), '--output', str(target),
                   '--shared-state', str(pool / 'shared_source_state.npz'),
                   '--topology-source-sha256', request['topology_source_sha256'],
                   '--gpu-builder-source-sha256', request['gpu_builder_source_sha256']]
        ledger[rid]['execution'] = 'running'
        snapshot()
        started = time.monotonic()
        with (output / (rid + '.log')).open('xb') as log:
            result = subprocess.run(command, cwd=runtime, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / (rid + '_exit.json'), {'command': command, 'exit_code': result.returncode,
                   'elapsed_s': time.monotonic() - started, 'completed_utc': utc_now()})
        record = {'run_id': rid, 'exit_code': result.returncode}
        if result.returncode:
            ledger[rid]['execution'] = 'failed'
            failures += 1
        else:
            ledger[rid]['execution'] = 'completed'
            record['binding'] = binding_check(target)
            record['reset'] = reset_check(pool / 'shared_source_state.npz', target / 'reset_returned.npz')
            try:
                clock = review_clock_schedule(target / 'clock_events.jsonl', target / 'actual_tools')
                write_json(output / (rid + '_clock.json'), clock)
                record['clock'] = {'status': clock['status']}
            except Exception as exc:  # recorded for review, not a stop-rule failure
                record['clock'] = {'status': 'failed', 'error': repr(exc)}
            try:
                diagnosis = diagnose_tool_schedule(target, reference_log, reference_tools, center,
                                                   comparison_parameters(row))
                verdict = evaluate(diagnosis['metrics'], request['tool_interface_contract'])
                write_json(output / (rid + '_tool_interface.json'), {'diagnosis': diagnosis, 'contract': verdict,
                           'comparison_parameters_mm_mm_deg': comparison_parameters(row)})
                record['tool_interface'] = {'status': verdict['status'],
                                            'worst_gated_position_mm': verdict['worst_gated_position_mm']}
            except Exception as exc:
                record['tool_interface'] = {'status': 'failed', 'error': repr(exc)}
            if record['reset']['status'] != 'passed':
                failures += 1
            if record['binding']['status'] != 'passed':
                failures += 1
                stopped = 'gpu_binding_failure:' + rid
        ledger[rid]['checks'] = 'recorded'
        results[rid] = record
        write_json(output / (rid + '_checks.json'), record)
        if not stopped and failures > 0.2 * len(runs):
            stopped = 'failure_fraction_exceeded_after:' + rid
        snapshot()
    write_json(output / 'run_status_final.json', ledger)
    executed = sum(1 for v in ledger.values() if v['execution'] in ('completed', 'failed'))
    return {'status': 'stopped_by_rule' if stopped else 'batch_collected_requires_review',
            'stop_reason': stopped, 'planned_runs': len(runs), 'executed_runs': executed,
            'failed_runs': failures, 'not_executed': [k for k, v in ledger.items() if v['execution'] == 'not_executed'],
            'results': results, 'new_Dyn_rollouts': 0, 'candidate_pool_released': False,
            'task_success': 'not_evaluated', 'model_alignment': 'unverified', 'primary_frame': 60, 'primary_label_s': 1.920}


def collect(request_path, output):
    output.mkdir(exist_ok=False)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    status, started = 0, time.monotonic()
    try:
        report = run(request, request_path.parent, output)
        if report['status'] != 'batch_collected_requires_review':
            status = 2
    except Exception as exc:
        status = 1
        report = {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(),
                  'candidate_pool_released': False}
        write(output / 'batch_failure.json', report)
    write(output / 'batch_report.json', report)
    write(output / 'request.json', request)
    write(output / 'stage_exit.json', {'exit_code': status, 'elapsed_s': time.monotonic() - started,
          'execution_subprocess_exits_recorded': len(list(output.glob('*_exit.json'))),
          'failed_or_interrupted_evidence_retained': True, 'candidate_pool_released': False})
    write(output / 'evidence_manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): digest(p)
          for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts}})
    with zipfile.ZipFile(output.parent / 'batch_evidence.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for p in output.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                archive.write(p, p.relative_to(output).as_posix())
    print(json.dumps({k: report.get(k) for k in ('status', 'stop_reason', 'planned_runs', 'executed_runs', 'failed_runs')}))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(collect(args.request.resolve(), args.output.resolve()))
