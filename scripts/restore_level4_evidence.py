"""CPU restoration from verified local bodies; no model/simulator or NSCC reads."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from level4.artifacts import read_json, write_json, sha256, load_grid, load_trajectory, utc_now
from level4.archive import verify_files
from level4.geometry import measure


def render_topview(points_mm, output, title):
    """Offline point-state rendering only; never changes metric geometry."""
    from PIL import Image, ImageDraw
    # MPM snapshots contain float32. Renderer metadata must use Python numbers,
    # while the original arrays remain untouched for all metric calculations.
    points = np.asarray(points_mm, dtype=float)
    image = Image.new('RGB', (800, 650), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((25, 15), title, fill='black')
    low, high = points[:, :2].min(0), points[:, :2].max(0)
    extent = np.maximum(high - low, 1.)
    scale = float(min(680 / extent[0], 500 / extent[1]))
    center = (low + high) / 2
    xy = (points[:, :2] - center) * [scale, -scale] + [400, 320]
    for x, y in xy:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(35, 85, 145))
    draw.line((60, 590, 140, 590), fill='black', width=2)
    draw.text((60, 605), 'scale bar: %.3f mm' % (80 / scale), fill='black')
    draw.text((25, 40), 'Top view: world x/y; raw points, no smoothing or hole filling', fill='black')
    image.save(output)
    return {'projection': 'orthographic_world_xy', 'pixels_per_mm': scale,
            'center_mm': center.tolist(), 'render_only_no_metric_modification': True}


def restore_synthetic(example, output):
    from level4.pipeline import prediction_decisions
    example, output = Path(example), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if read_json(example / 'WARNING.json').get('kind') == 'real_experiment':
        raise ValueError('Synthetic restore cannot claim real experiment')
    expected = read_json(example / 'selection.json')
    spec = read_json(example / 'spec.json')
    reports, decisions, _ = prediction_decisions(example / 'pool', spec, experiment=False)
    if decisions != expected['selections']:
        raise ValueError('Restored decisions differ from saved synthetic selection')
    write_json(output / 'evaluations.json', reports)
    write_json(output / 'decisions.json', decisions)
    grid = load_grid(example / 'pool')
    trajectory = load_trajectory(example / 'pool/predictions/candidate_000/trajectory.npz', grid)
    with np.load(example / 'pool/predictions/candidate_000/trajectory.npz', allow_pickle=False) as packet:
        points = grid['query'][packet['classes'][-1] > 0] * float(grid['unit_to_meter']) * 1000
    camera = render_topview(points, output / 'restored_state.png', 'SYNTHETIC ONLY - candidate_000 predicted final state')
    write_json(output / 'camera.json', camera)
    with (output / 'spans.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['time_s', 'u_mm', 'v_mm'])
        for t, geometry in trajectory:
            writer.writerow([t, measure(geometry, [1, 0, 0])['value'], measure(geometry, [0, 1, 0])['value']])
    report = {'kind': 'restoration_test', 'status': 'passed', 'scope': 'synthetic_only',
              'created_utc': utc_now(), 'matches_original_decisions': True,
              'server_access_used': False, 'new_simulation_run': False,
              'scientific_claim': 'No real run restored; no material success evidence',
              'original_selection_sha256': sha256(example / 'selection.json'),
              'output_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'restore_report.json', report)
    return report


def restore_reset(evidence, output):
    """Use archived NPZ initial bodies; no unsafe pickle load or original paths."""
    from scripts.audit_level4_reset_failure import initial_difference
    evidence, output = Path(evidence), Path(output)
    saved = read_json(evidence / 'report.json')
    hashes = read_json(evidence / 'evidence_manifest.json')['files_sha256']
    for name, digest in hashes.items():
        from level4.archive import safe_name
        safe_name(name)
        if sha256(evidence / name) != digest:
            raise ValueError('Archived evidence changed: ' + name)
    scale = saved['initial_difference']['unit_to_meter']
    tol = saved['initial_difference']['tolerance_mm_unchanged']
    with np.load(evidence / 'initial_states.npz', allow_pickle=False) as packet:
        reference, actual = packet['planner_initial_positions'], packet['execution_frame0_pos']
        compared = initial_difference(reference, actual, scale, tol)
    for key in ('reference_count', 'actual_count', 'strict_initial_check_passed', 'index_paired_max_abs_error_mm',
                'index_paired_component_rms_mm', 'reference_world_xyz_spans_mm', 'actual_world_xyz_spans_mm'):
        if not np.allclose(compared[key], saved['initial_difference'][key], rtol=1e-12, atol=1e-12):
            raise ValueError('Restored reset diagnostic differs: ' + key)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'initial_difference.json', compared)
    camera = render_topview(actual * scale * 1000, output / 'restored_state.png', 'Independent MPM execution - raw initial material')
    write_json(output / 'camera.json', camera)
    real_data = saved.get('kind') == 'offline_reset_failure_diagnosis_not_task_result'
    report = {'kind': 'restoration_test', 'status': 'passed',
              'scope': 'real_reset_diagnostic_only' if real_data else 'synthetic_reset_fixture_only',
              'created_utc': utc_now(), 'matches_original_summary': True, 'scientific_status': 'reset_failed',
              'server_access_used': False, 'new_simulation_run': False,
              'full_pipeline_restored': False, 'missing_for_full_pipeline': ['weights', 'grid', 'planner_input', 'runtime_file_bodies'],
              'input_manifest_sha256': sha256(evidence / 'evidence_manifest.json'),
              'output_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / 'restore_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('synthetic', 'reset'), required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = (restore_synthetic if args.kind == 'synthetic' else restore_reset)(args.input, args.output)
    print(__import__('json').dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
