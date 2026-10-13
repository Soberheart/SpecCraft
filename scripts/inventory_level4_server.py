"""Dependency-first NSCC inventory only; no qsub, training, simulation or copying."""
import argparse
from pathlib import Path
import shutil
import socket
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from level4.artifacts import read_json, write_json, utc_now
from level4.archive import inventory

ROOT = Path('/home/users/ntu/mingjie0/projects/doughnet')
RUNS = {
    '25768191': 'level4_nominal_request_UqDMa7',
    '25768683': 'level4_nominal_retry_W2tfbV',
    '25768898': 'level4_nominal_no_git_rwru1k1c',
    '25769075': 'level4_environment_probe_g0gwyxe6',
    '25769308': 'level4_nominal_reviewed_compute_8lqohq3_',
}


def collect(root, output, hash_files=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    commands = {}
    for label, argv in [('jobs', ['qstat', '-u', 'mingjie0'])] + [
            (job, ['qstat', '-xf', job + '.pbs101']) for job in RUNS]:
        try:
            result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            text = result.stdout.decode('utf-8', errors='replace')
            commands[label] = {'argv': argv, 'exit_code': result.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            text = str(exc)
            commands[label] = {'argv': argv, 'exit_code': None, 'error': str(exc)}
        (output / ('qstat_' + label + '.txt')).write_text(text, encoding='utf-8')
    sources = [{'id': 'job_' + job, 'path': str(root / 'records' / name), 'category': 'level4_run'}
               for job, name in RUNS.items()]
    dyn = root / 'records/level3_pointcloud/sim_check_FhY2DG/dyn_check_ml9xhm/artifacts'
    sources.extend([
        {'id': 'level3_dyn_input', 'path': str(dyn), 'category': 'input'},
        {'id': 'checkpoint', 'path': str(root / 'outputs/2026-10-04/14-59-24/best.pth'), 'category': 'weights',
         'expected_sha256': 'a05452f681f89ee26763959ec77db234dc0bdcf2cbd2996f7361beebb563acae'},
        {'id': 'model_hydra', 'path': str(root / 'outputs/2026-10-04/14-59-24/.hydra'), 'category': 'config'},
        {'id': 'reset_diagnosis', 'path': str(root / 'records/level4_initial_reset_diagnosis_mvrcvqpw/evidence'), 'category': 'real_reset_evidence'},
        {'id': 'preflight', 'path': str(root / 'records/level4_server_preflight_4v7w8ej1'), 'category': 'preflight'},
    ])
    metadata = read_json(dyn / 'metadata.json') if (dyn / 'metadata.json').is_file() else {}
    # This is an explicit dependency in a verified-input metadata file; retain bodies, not links.
    if metadata.get('scene_dir'):
        sources.append({'id': 'source_scene', 'path': metadata['scene_dir'], 'category': 'initial_reference_raw'})
    request = {'run': {'run_id': 'level4_server_dependency_inventory', 'level': 4, 'task': 'backup_audit',
                       'method': 'inventory_only', 'data_kind': 'mixed', 'scientific_status': 'unverified',
                       'stability_basis': None, 'requirements_checked': False,
                       'state_note': 'Per-job qstat evidence saved; do not copy until completed/failed stable sources are reviewed.',
                       'independent_backup': None}, 'sources': sources}
    write_json(output / 'request.json', request)
    manifest = inventory(request, hash_files)
    write_json(output / 'manifest.json', manifest)
    summary = {'generated_utc': utc_now(), 'host': socket.gethostname(), 'root': str(root),
               'mode': 'hashed_inventory' if hash_files else 'capacity_metadata_only',
               'simulation_run': False, 'training_run': False, 'source_modified': False,
               'filesystem_free_bytes': shutil.disk_usage(root).free,
               'totals': manifest['totals'], 'sources': [], 'gaps': manifest['gaps'], 'qstat_commands': commands}
    for source in sources:
        rows = [v for k, v in manifest['files'].items() if k.startswith(source['id'] + '/')]
        summary['sources'].append({'id': source['id'], 'path': source['path'],
                                   'files': len(rows), 'bytes': sum(v['size'] for v in rows)})
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hash', action='store_true', help='After capacity review; may read large files')
    args = parser.parse_args()
    print(__import__('json').dumps(collect(args.root, args.output, args.hash), indent=2))


if __name__ == '__main__': main()
