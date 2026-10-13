"""Offline CPU diagnosis of retained reset failures; no acceptance-rule changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import zipfile
import numpy as np

DEFAULT_ROOT = Path('/home/users/ntu/mingjie0/projects/doughnet')
DEFAULT_PARENT = DEFAULT_ROOT / 'records/level4_nominal_reviewed_compute_8lqohq3_'
DEFAULT_POOL = DEFAULT_PARENT / 'level4_development_9RtTk7/reset_1/pool'
DEFAULT_ARCHIVE = DEFAULT_ROOT / 'records/level4_initial_reset_diag_20261012_01.zip'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def checked_particles(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError('Material positions must be nonempty finite Nx3 arrays')
    return array


def initial_difference(reference, actual, unit_to_meter, tolerance_mm):
    """Report index-paired differences; set/order diagnostics never override them."""
    reference, actual = checked_particles(reference), checked_particles(actual)
    if (isinstance(unit_to_meter, bool) or not np.isfinite(unit_to_meter) or unit_to_meter <= 0
            or isinstance(tolerance_mm, bool) or not np.isfinite(tolerance_mm) or tolerance_mm <= 0):
        raise ValueError('Scale and provisional reset tolerance must be positive finite numbers')
    factor = 1000 * float(unit_to_meter)
    result = {
        'reference_count': len(reference), 'actual_count': len(actual),
        'unit_to_meter': float(unit_to_meter), 'tolerance_mm_unchanged': float(tolerance_mm),
        'reference_bbox_mm': [list(reference.min(0) * factor), list(reference.max(0) * factor)],
        'actual_bbox_mm': [list(actual.min(0) * factor), list(actual.max(0) * factor)],
        'reference_centroid_mm': list(reference.mean(0) * factor),
        'actual_centroid_mm': list(actual.mean(0) * factor),
        'centroid_shift_mm': list((actual.mean(0) - reference.mean(0)) * factor),
        'reference_world_xyz_spans_mm': list(np.ptp(reference, axis=0) * factor),
        'actual_world_xyz_spans_mm': list(np.ptp(actual, axis=0) * factor),
        'index_paired_max_abs_error_mm': None, 'strict_initial_check_passed': False,
        'index_paired_component_rms_mm': None, 'index_paired_max_abs_per_axis_mm': None,
        'index_paired_distance_quantiles_mm': None, 'lexicographic_max_abs_error_mm': None,
        'diagnostic_scope': 'offline only; nearest-neighbor or reordered geometry cannot certify reset equality'}
    if reference.shape != actual.shape:
        result['reason'] = 'particle_count_or_shape_mismatch'
        return result
    difference = actual - reference
    maximum_units = float(np.max(np.abs(difference)))
    distances = np.linalg.norm(difference, axis=1) * factor
    order_reference = np.lexsort(reference.T[::-1])
    order_actual = np.lexsort(actual.T[::-1])
    result.update(
        index_paired_max_abs_error_mm=maximum_units * factor,
        index_paired_component_rms_mm=float(np.sqrt(np.mean(difference ** 2)) * factor),
        index_paired_max_abs_per_axis_mm=list(np.max(np.abs(difference), axis=0) * factor),
        index_paired_distance_quantiles_mm=dict(zip(('q0', 'q50', 'q90', 'q99', 'q100'),
            map(float, np.quantile(distances, [0, .5, .9, .99, 1])))),
        lexicographic_max_abs_error_mm=float(np.max(np.abs(actual[order_actual] - reference[order_reference])) * factor),
        strict_initial_check_passed=maximum_units <= tolerance_mm / factor,
        reason='within_existing_index_paired_limit' if maximum_units <= tolerance_mm / factor else 'index_paired_limit_exceeded')
    return result


def frame_schedule(logs):
    if not isinstance(logs, list) or not logs:
        raise ValueError('Expected a nonempty trusted simulator log list')
    result = [{'index': i, 'step': int(row['step']), 't': float(row['t']),
               'frame_idx': int(row['frame_idx']) if 'frame_idx' in row else None,
               'particle_count': len(checked_particles(row['obj']['particles']['pos']))}
              for i, row in enumerate(logs)]
    if not np.isfinite([row['t'] for row in result]).all():
        raise ValueError('Nonfinite raw log times')
    return result


def configuration_differences(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            name = prefix + '.' + key if prefix else key
            rows.extend(configuration_differences(before.get(key), after.get(key), name))
        return rows
    if before == after:
        return []
    return [{'path': prefix, 'before': before, 'after': after}]


def audit(pool, output, archive):
    pool, output, archive = Path(pool).resolve(), Path(output).resolve(), Path(archive).resolve()
    if output == pool or pool in output.parents or archive == pool or pool in archive.parents:
        raise ValueError('Write diagnostic evidence outside the original pool')
    if output.exists() or archive.exists():
        raise FileExistsError('Use new diagnostic output and archive paths; retain prior evidence')
    manifest = json.loads((pool / 'pool.json').read_text(encoding='utf-8'))
    for name, expected in manifest['input_sha256'].items():
        if digest(pool / name) != expected:
            raise ValueError('Retained pool input changed: ' + name)
    execution = pool / 'executions/candidate_000'
    exit_record = json.loads((execution / 'exit.json').read_text(encoding='utf-8'))
    if exit_record['pool_sha256'] != digest(pool / 'pool.json') or exit_record['exit_code'] != 0:
        raise ValueError('Diagnosis requires the retained pool and a zero simulator process exit')
    metadata = json.loads((pool / 'source_metadata.json').read_text(encoding='utf-8'))
    source_scene = Path(metadata['scene_dir'])
    source_log, actual_log = source_scene / 'log.pkl', execution / 'scenes/0000/log.pkl'
    if (digest(source_log) != metadata['source_log_sha256']
            or digest(source_scene / 'config.yaml') != metadata['scene_config_sha256']):
        raise ValueError('Original source scene/log changed')
    with source_log.open('rb') as stream:
        source = pickle.load(stream)
    with actual_log.open('rb') as stream:
        actual = pickle.load(stream)
    source_schedule, actual_schedule = frame_schedule(source), frame_schedule(actual)
    with np.load(pool / 'planner_input.npz', allow_pickle=False) as packet:
        reference = checked_particles(packet['initial_particles'])
    with np.load(pool / 'grid.npz', allow_pickle=False) as packet:
        scale = float(packet['unit_to_meter'])
    original_positions = checked_particles(source[0]['obj']['particles']['pos'])
    if not np.array_equal(original_positions, reference):
        raise ValueError('Planner initial material reference differs from source log frame0')
    tolerance = float(manifest['config']['initial_reset_max_error_mm'])
    compared = initial_difference(reference, actual[0]['obj']['particles']['pos'], scale, tolerance)
    # Nearest-neighbor shape similarity diagnoses possible permutation/resampling;
    # it cannot replace the strict existing index-paired reset gate.
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        compared['nearest_neighbor'] = {'status': 'not_run', 'reason': 'scipy unavailable'}
    else:
        current = checked_particles(actual[0]['obj']['particles']['pos'])
        distances_a = cKDTree(reference).query(current)[0] * 1000 * scale
        distances_b = cKDTree(current).query(reference)[0] * 1000 * scale
        compared['nearest_neighbor'] = {'status': 'diagnostic_only',
            'symmetric_max_mm': float(max(distances_a.max(), distances_b.max())),
            'actual_to_reference_q50_mm': float(np.median(distances_a)),
            'reference_to_actual_q50_mm': float(np.median(distances_b))}
    arrays = {'planner_initial_positions': reference}
    for label, logs in (('source', source), ('execution', actual)):
        for i, row in enumerate(logs[:2]):
            for key in ('pos', 'vel'):
                if key in row['obj']['particles']:
                    arrays[label + '_frame' + str(i) + '_' + key] = np.asarray(row['obj']['particles'][key])
        arrays[label + '_times_s'] = np.asarray([row['t'] for row in logs])
        arrays[label + '_steps'] = np.asarray([row['step'] for row in logs])
    report = {'kind': 'offline_reset_failure_diagnosis_not_task_result',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'pool': str(pool),
              'candidate_id': 'candidate_000', 'simulator_exit': exit_record,
              'initial_difference': compared, 'source_schedule': source_schedule,
              'execution_schedule': actual_schedule,
              'simulator_seed': manifest['simulator_seed'],
              'observation_seed': manifest['observation_seed'],
              'nominal_action': next(row for row in manifest['candidates'] if row['candidate_id'] == 'candidate_000'),
              'initial_source': 'source log frame0; both old logs record after scene.step at label t=0',
              'new_simulation_run': False, 'reset_gate_changed': False,
              'configuration_text': {}, 'source_files': {}}
    try:
        from omegaconf import OmegaConf
    except ImportError:
        report['configuration_comparison'] = {'status': 'not_run', 'reason': 'omegaconf unavailable; exact YAML files retained'}
    else:
        configs = [OmegaConf.to_container(OmegaConf.load(path), resolve=True) for path in (
            pool / 'source_scene.yaml', pool / 'candidate_000.yaml', execution / 'scenes/0000/config.yaml')]
        report['configuration_comparison'] = {'status': 'compared',
            'source_to_candidate': configuration_differences(configs[0], configs[1]),
            'candidate_to_actual': configuration_differences(configs[1], configs[2])}
    output.mkdir(parents=True, exist_ok=False)
    files = {
        'pool.json': pool / 'pool.json', 'pool_config.json': pool / 'pool_config.json',
        'source_metadata.json': pool / 'source_metadata.json',
        'source_scene.yaml': pool / 'source_scene.yaml', 'candidate_000.yaml': pool / 'candidate_000.yaml',
        'actual_scene.yaml': execution / 'scenes/0000/config.yaml',
        'source_log.pkl': source_log, 'execution_log.pkl': actual_log,
        'exit.json': execution / 'exit.json', 'generate.log': execution / 'generate.log',
        'runtime_snapshot_manifest.json': pool / 'runtime_snapshot/manifest.json'}
    if (execution / 'failure.json').exists():
        files['failure.json'] = execution / 'failure.json'
    # For the default retained server run, preserve scheduler, environment and
    # submit-manifest evidence without copying entire source snapshots again.
    run = pool.parents[1]
    request = run.parent
    for name in ('exit_code.txt', 'runtime_environment_check.json', 'nvidia-smi.txt', 'environment.txt',
                 'qstat_start.txt', 'qstat_end.txt', 'snapshot.json', 'run_location.txt'):
        if (run / name).is_file():
            files['compute_' + name] = run / name
    for name in ('qstat_finished.txt', 'qstat_finished_exit_code.txt', 'pbs.log', 'job_id.txt',
                 'runtime/manifest.json', 'execution_environment_probe.json'):
        if (request / name).is_file():
            files['request_' + name.replace('/', '_')] = request / name
    files['collector.py'] = Path(__file__).resolve()
    for name, source_path in files.items():
        before = digest(source_path)
        shutil.copyfile(source_path, output / name)
        if digest(source_path) != before or digest(output / name) != before:
            raise ValueError('Source changed during diagnostic archival: ' + str(source_path))
        report['source_files'][name] = {'path': str(source_path), 'sha256': before}
        if name.endswith('.yaml'):
            report['configuration_text'][name] = source_path.read_text(encoding='utf-8')
    with (output / 'initial_states.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report['initial_states_sha256'] = digest(output / 'initial_states.npz')
    write_json(output / 'report.json', report)
    write_json(output / 'evidence_manifest.json', {
        'kind': 'offline_reset_failure_archive', 'not_a_simulation_or_reset_acceptance': True,
        'files_sha256': {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
    result = {'status': 'diagnostic_evidence_saved', 'initial_difference': compared,
              'source_frame_count': len(source), 'execution_frame_count': len(actual),
              'source_first_frame': source_schedule[0], 'execution_first_frame': actual_schedule[0],
              'source_last_frame': source_schedule[-1], 'execution_last_frame': actual_schedule[-1],
              'configuration_comparison': report['configuration_comparison'],
              'output': str(output), 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest(archive), 'new_simulation_run': False, 'reset_gate_changed': False}
    write_json(output / 'archive_receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=DEFAULT_POOL)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    if args.archive.exists():
        raise FileExistsError('Archive already exists; retain it and specify a new --archive')
    if args.output is None:
        base = Path(tempfile.mkdtemp(prefix='level4_initial_reset_diagnosis_', dir=DEFAULT_ROOT / 'records'))
        args.output = base / 'evidence'
    print(json.dumps(audit(args.pool, args.output, args.archive), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('RESET_DIAGNOSIS_FAILED: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
