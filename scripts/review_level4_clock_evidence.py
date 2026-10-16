"""Restore a collected clock diagnostic in a new CPU directory.

Run from a reconstructed, hash-checked runtime. This does not invoke SSH,
the simulator, a model, selection, or threshold calibration.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.artifacts import read_json, write_json, sha256, utc_now
from level4.archive import unpack_archive, verify_zip, safe_name
from level4.candidate_interface import diagnose_tool_schedule
from level4.clock_review import review_clock_schedule
from level4.geometry import Geometry, measure
from level4.initial_frame import assert_same_frame
from level4.shared_input import exact_numeric_packet


def csv_rows(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recover(collection, output, expected_sha256, plot_dependencies=None):
    collection, output = Path(collection).resolve(), Path(output).resolve()
    header = read_json(collection / 'transfer_header.json')
    receipt = read_json(collection.parent / 'submission.json')
    if (header['job'] != receipt['job'] or header['remote_parent'] != receipt['remote_parent']
            or not re.search(r'job_state\s*=\s*F\b', header['qstat'])
            or not re.search(r'Exit_status\s*=\s*0\b', header['qstat'])):
        raise ValueError('Expected a completed diagnostic matching its submission')
    names = [row['name'] for row in header['files']]
    if len(set(names)) != len(names) or 'diagnostic_evidence.zip' not in names:
        raise ValueError('Invalid transfer file manifest')
    for row in header['files']:
        safe_name(row['name'])
        path = collection / row['name']
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Transferred evidence differs: ' + row['name'])
    if (collection / 'job_exit_code.txt').read_text().strip() != '0':
        raise ValueError('Job exit file differs')
    archive = collection / 'diagnostic_evidence.zip'
    internal = verify_zip(archive)
    output.mkdir(parents=True, exist_ok=False)
    unpack = unpack_archive(archive, output / 'evidence', expected_sha256, max_bytes=512 * 1024**2)
    evidence = output / 'evidence'
    request = read_json(evidence / 'request.json')
    if (request['kind'] != 'one_repeat_candidate001_clock_diagnostic_v1'
            or request['candidate_ids'] != ['candidate_001']
            or request['new_simulations_max'] != 1 or request['new_Dyn_rollouts'] != 0
            or request['candidate_pool_released'] is not False):
        raise ValueError('Unexpected diagnostic scope')
    if sha256(evidence / 'submitted_toolkit.zip') != receipt['toolkit_sha256']:
        raise ValueError('Frozen toolkit identity differs')
    stage = read_json(evidence / 'stage_exit.json')
    simulator = read_json(evidence / 'candidate_001_exit.json')
    if stage['exit_code'] != 0 or simulator['exit_code'] != 0 or stage['candidate_pool_released'] is not False:
        raise ValueError('Diagnostic exit or gate state differs')
    pool_path = evidence / 'source_pool_manifest.json'
    if sha256(pool_path) != request['source_pool_sha256']:
        raise ValueError('Frozen pool identity differs')
    pool = read_json(pool_path)
    action = next(row for row in pool['candidates'] if row['candidate_id'] == 'candidate_001')
    candidate, reference = evidence / 'execution/candidate_001', evidence / 'nominal_reference'
    exact_numeric_packet(reference / 'reset_returned.npz', candidate / 'reset_returned.npz')
    exact_numeric_packet(reference / 'static_material_info.npz', candidate / 'static_material_info.npz')
    with np.load(candidate / 'after_first_action.npz', allow_pickle=False) as first, np.load(
            reference / 'reset_returned.npz', allow_pickle=False) as initial:
        for key in ('mpm_pos', 'mpm_vel', 'mpm_C', 'mpm_F', 'mpm_active',
                    'tool_000_pos', 'tool_000_quat', 'tool_000_vel', 'tool_000_ang'):
            if not np.array_equal(first[key], initial[key]):
                raise ValueError('First action changed material or ground')
    with (candidate / 'scenes/0000/log.pkl').open('rb') as stream:
        actual = pickle.load(stream)
    if len(actual) != 63:
        raise ValueError('Incomplete material log')
    for frame, row in enumerate(actual):
        with (candidate / 'raw_frames' / ('frame_%03d.pkl' % frame)).open('rb') as stream:
            assert_same_frame(row, pickle.load(stream))
    # This frozen scene has nominal grasp center (0,0,0); reject other scenes.
    with (reference / 'legacy_log.pkl').open('rb') as stream:
        legacy = pickle.load(stream)
    center = np.zeros(3)
    if not np.array_equal(legacy[0]['ee']['pos'], center):
        raise ValueError('Expected the frozen zero-center scene')
    tools = diagnose_tool_schedule(candidate, reference / 'legacy_log.pkl', reference / 'actual_tools',
                                   center, action['parameters_mm_mm_deg'])
    clock = review_clock_schedule(candidate / 'clock_events.jsonl', candidate / 'actual_tools')
    if tools != read_json(evidence / 'tool_diagnosis.json'):
        raise ValueError('Offline tool diagnosis differs from the recorded report')
    if clock['frozen_replay'] != read_json(evidence / 'clock_integration_report.json'):
        raise ValueError('Offline event replay differs from the recorded report')
    axes = pool['config']['axes']
    spans = []
    for frame, row in enumerate(actual):
        geometry = Geometry(row['obj']['particles']['pos'], .2, 'points', True, True)
        values = [measure(geometry, axes[key]) for key in ('u', 'v')]
        if any(value['status'] != 'measured' for value in values):
            raise ValueError('Invalid raw particle center geometry')
        spans.append({'frame': frame, 'original_label_s': float(row['t']),
                      'retained_elapsed_s_from_frame0': clock['records'][frame]['active_branch_elapsed_s'],
                      'raw_particle_center_u_mm': values[0]['value'], 'raw_particle_center_v_mm': values[1]['value']})
    with np.load(reference / 'original_planner_input.npz', allow_pickle=False) as initial:
        initial_error = float(np.max(np.abs(actual[0]['obj']['particles']['pos'].astype(float)
                                            - initial['initial_particles'].astype(float))) * 200)
    if initial_error > .001:
        raise ValueError('Original logged0 reset limit exceeded')
    write_json(output / 'clock_review.json', clock)
    write_json(output / 'tool_diagnosis.json', tools)
    csv_rows(output / 'clock_timeline.csv', clock['records'])
    csv_rows(output / 'tool_errors.csv', tools['records'])
    csv_rows(output / 'raw_material_spans.csv', spans)
    write_json(output / 'raw_material_summary.json', {
        'kind': 'diagnostic_particle_center_extents_not_primary_voxel_metrics', 'unit': 'mm', 'axes': axes,
        'frame0': spans[0], 'original_main_frame60': spans[60], 'auxiliary_frame62': spans[62],
        'task_satisfaction': 'not_evaluated_uncalibrated', 'no_smoothing_or_hole_filling': True,
        'original_labels_unchanged': True, 'new_primary_time_chosen': False})
    if plot_dependencies is not None:
        sys.path.insert(0, str(Path(plot_dependencies).resolve()))
    os.environ['MPLCONFIGDIR'] = str(output / 'plot_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, panels = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True, layout='constrained')
    for panel, frame in zip(panels, (0, 60)):
        points = actual[frame]['obj']['particles']['pos'].astype(float) * 200
        panel.scatter(points[:, 0], points[:, 1], s=1.4, color='#007f71', rasterized=True)
        panel.set_title('frame%d: label %.3fs; retained %.3fs' % (frame, actual[frame]['t'], spans[frame]['retained_elapsed_s_from_frame0']))
        panel.set(xlabel='world x (mm)', ylabel='world y (mm)', xlim=(-50, 50), ylim=(-50, 50))
        panel.set_aspect('equal')
    fig.suptitle('candidate_001 raw MPM particles; CPU recovery; task not evaluated', fontsize=10)
    fig.savefig(output / 'raw_material_frame0_frame60.png', dpi=160)
    fig.savefig(output / 'raw_material_frame0_frame60.pdf')
    plt.close(fig)
    fig, panels = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout='constrained')
    frames = np.arange(63)
    panels[0].plot(frames, [row['elapsed_minus_label_s'] * 1000 for row in clock['records']])
    panels[0].set_ylabel('Retained time minus label (ms)')
    panels[1].plot(frames, [row['finger_1_physical_transform_error_mm'] for row in tools['records']], label='left finger vs nominal transform')
    panels[1].plot(frames, [row['summary_opening_error_mm'] for row in tools['records']], label='opening vs nominal')
    panels[1].axhline(.001, ls='--', color='black', label='original 0.001mm gate')
    panels[1].set(xlabel='Original saved frame index', ylabel='Difference (mm)')
    panels[1].legend(fontsize=8)
    for panel in panels:
        panel.axvline(60, color='gray', ls=':')
        panel.grid(alpha=.2)
    fig.suptitle('Clock mismatch and tool differences; nominal reference clock unavailable', fontsize=10)
    fig.savefig(output / 'clock_and_tool_diagnostic.png', dpi=160)
    fig.savefig(output / 'clock_and_tool_diagnostic.pdf')
    plt.close(fig)
    write_json(output / 'camera.json', {'projection': 'world_xy', 'bounds_mm': [-50, 50, -50, 50],
                                       'frames': [0, 60], 'geometry': 'raw_material_particle_centers', 'unit_to_meter': .2})
    report = {'created_utc': utc_now(), 'status': 'passed_real_clock_diagnostic_CPU_restore_not_task_success',
              'job': header['job'], 'job_exit_code': 0, 'simulator_exit_code': 0, 'simulator_elapsed_s': simulator['elapsed_s'],
              'transfer_files_verified': len(names), 'internal_verification': internal, 'unpack_verification': unpack,
              'original_NSCC_paths_used': False, 'scientific_runtime': str(REPO),
              'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__, 'python': sys.version,
              'complete_reset_and_static_material_exact': True, 'first_action_material_and_ground_exact': True,
              'logged0_material_max_abs_error_mm': initial_error, 'original_reset_limit_mm_unchanged': .001,
              'raw63frames_all_nested_fields_dtype_shape_value_identical': True,
              'frozen_event_report_and_all_tool_records_reproduced_exactly': True,
              'event_count': clock['frozen_replay']['events'], 'clock_status': clock['status'],
              'main_frame60': clock['main_frame60'], 'tool_transform_status': tools['tool_transform_status'],
              'nominal_reference_physical_clock': 'not_recorded_not_certified',
              'legacy_model_tool_input_alignment': 'not_certified', 'task_satisfaction': 'not_evaluated_uncalibrated',
              'candidate_pool_released': False, 'new_simulations': 0, 'model_prediction_run': False,
              'full_GPU_environment_restore': 'pending', 'archive_sha256': expected_sha256,
              'output_hashes': {path.name: sha256(path) for path in output.iterdir() if path.is_file()}}
    write_json(output / 'restore_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--plot-dependencies', type=Path)
    args = parser.parse_args()
    result = recover(args.collection, args.output, args.expected_sha256, args.plot_dependencies)
    print(json.dumps({key: result[key] for key in ('status', 'job', 'event_count', 'clock_status', 'tool_transform_status', 'new_simulations')}))
