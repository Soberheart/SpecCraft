"""L4-01 Phase B job preparation and submission on the NSCC login node (CPU only).

Run from the git checkout on NSCC after ``git pull --ff-only``:

    python scripts/level4_phase_b.py plan
    python scripts/level4_phase_b.py prepare B-pilot --submit
    python scripts/level4_phase_b.py status

``prepare`` builds a new job directory under the batch root with ``tools/``
(frozen job-22 execution sources, verified by hash, plus the new runners),
``request.json``, the PBS script and a login-node verification receipt; with
``--submit`` it also calls ``qsub`` and records the job id.  Nothing here
touches a GPU or runs a simulation.  Execution jobs B-exec-* are refused until
the pilot's collected report passes every per-run check; B-S4 needs the
B-exec reports and an explicit, rule-based candidate choice.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/users/ntu/mingjie0/projects/doughnet')
BATCH_DIRNAME = 'records/level4_discriminative_batch_20261016'
FROZEN22_DIRNAME = 'records/level4_shared_reset_c0bb58e9eedd4f55b633da21cdb04f50'
FROZEN22_REQUEST_SHA256 = 'c27c44bcfd50407e59b078c8ccdb69d87dfe86e106e6d16a472b451265b6e54a'
FROZEN22_TOOL_MANIFEST_SHA256 = '4a448124702bc5258604a8d2830c14b039dea013bf535aac676278db279c2a44'
CONFIG = REPO / 'configs/level4/l4_01'
QSUB_SELECT = 'select=1:ngpus=1:ncpus=8:mem=64GB'
FROZEN_REUSED = ('prepare_level4_shared_input.py', 'resume_level4_shared_input.py',
                 'probe_level4_initialization.py', 'probe_level4_candidate_interface.py')
NEW_OVERLAYS = {'level4/tool_interface_contract.py': 'l4_tool_interface_contract.py',
                'level4/tool_observation.py': 'l4_tool_observation.py'}
COPIED_FROM_REQUEST = ('runtime', 'runtime_manifest', 'runtime_overlays', 'required_files',
                       'expected_compute_environment', 'topology_source_sha256', 'gpu_binding_policy',
                       'gpu_builder_source_sha256', 'source_pool', 'source_pool_sha256', 'execution_version',
                       'legacy_source_log')
PILOT = ('candidate_002', 'candidate_016', 'candidate_032', 'S1_x_0p02')
SENSITIVITY_IN_EXEC = ('S0_repeat_nominal', 'S1_y_0p02', 'S2_x_0p1')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_plan(config=None):
    """Frozen run registry from the committed pool32/sensitivity manifests."""
    config = CONFIG if config is None else config
    pool32 = read(config / 'pool32_manifest.json')
    sens = read(config / 'sensitivity_manifest.json')
    runs = {}
    for row in pool32['candidates']:
        if row['phase_b_plan'] == 'execute_and_predict':
            runs[row['candidate_id']] = {'run_id': row['candidate_id'], 'candidate_id': row['candidate_id'],
                                         'group': row['group'], 'action_mm_mm_deg': row['parameters_mm_mm_deg'],
                                         'tool_offset_mm': [0.0, 0.0], 'scene': row['scene'],
                                         'scene_sha256': row['scene_sha256'],
                                         'scene_origin': 'frozen_pool' if row['group'] == 'frozen_seed0' else 'repo'}
    for row in sens['runs']:
        runs[row['run_id']] = {'run_id': row['run_id'], 'candidate_id': row['source_candidate'], 'group': 'sensitivity',
                               'action_mm_mm_deg': row['action_mm_mm_deg'], 'tool_offset_mm': row['tool_offset_mm'],
                               'scene': row['scene'], 'scene_sha256': row['scene_sha256'], 'scene_origin': 'repo'}
    remaining = [r for r in runs if r not in PILOT and r not in SENSITIVITY_IN_EXEC]
    remaining = sorted(remaining) + list(SENSITIVITY_IN_EXEC)
    sizes, jobs, start = (9, 8, 8, 8), {'B-pilot': list(PILOT)}, 0
    for index, size in enumerate(sizes, 1):
        jobs['B-exec-%d' % index] = remaining[start:start + size]
        start += size
    if start != len(remaining) or len(runs) != 37:
        raise ValueError('Frozen plan expects 33 pool + 4 sensitivity runs')
    predict = [{'candidate_id': r['candidate_id'], 'action_mm_mm_deg': r['parameters_mm_mm_deg']}
               for r in pool32['candidates'] if r['phase_b_plan'] in ('execute_and_predict', 'existing_execution_reuse')]
    return {'runs': runs, 'jobs': jobs, 'predict_candidates': predict, 'pool32': pool32}


def frozen22(project):
    parent = project / FROZEN22_DIRNAME
    request_path, manifest_path = parent / 'tools/request.json', parent / 'tools/tool_manifest.json'
    if sha256(request_path) != FROZEN22_REQUEST_SHA256 or sha256(manifest_path) != FROZEN22_TOOL_MANIFEST_SHA256:
        raise ValueError('Frozen job-22 request/tool manifest on NSCC differs from the reviewed copy')
    manifest = read(manifest_path)['files_sha256']
    for name, digest in manifest.items():
        if sha256(parent / 'tools' / name) != digest:
            raise ValueError('Frozen job-22 tool changed: ' + name)
    return parent, read(request_path), manifest


def file_entry(path):
    path = Path(path)
    return {'size': path.stat().st_size, 'sha256': sha256(path)}


def build_tools(parent, frozen_parent, frozen_request, frozen_manifest, runner):
    tools = parent / 'tools'
    tools.mkdir()
    for tool_name in set(frozen_request['runtime_overlays'].values()) | set(FROZEN_REUSED):
        if tool_name not in frozen_manifest:
            raise ValueError('Frozen tool missing from manifest: ' + tool_name)
        shutil.copyfile(frozen_parent / 'tools' / tool_name, tools / tool_name)
    for name, tool_name in NEW_OVERLAYS.items():
        shutil.copyfile(REPO / name, tools / tool_name)
    shutil.copyfile(REPO / 'scripts' / runner, tools / runner)
    return tools


def base_request(frozen_request, kind, label):
    request = {key: frozen_request[key] for key in COPIED_FROM_REQUEST}
    request['runtime_overlays'] = dict(frozen_request['runtime_overlays'], **NEW_OVERLAYS)
    request.update(kind=kind, job_label=label, created_utc=datetime.now(timezone.utc).isoformat(),
                   task_sheet='L4-01', decision='D-1', frozen_reference_request_sha256=FROZEN22_REQUEST_SHA256,
                   repo_commit=git_head(), frozen_plan_document={
                       'path': 'docs/level4_l4_01_next_minimal_experiment.md',
                       'sha256': sha256(REPO / 'docs/level4_l4_01_next_minimal_experiment.md')},
                   candidate_pool_released=False, main_frame=60, main_s=1.92,
                   auxiliary_frame=62, auxiliary_s=1.984, thresholds='null_development_only')
    return request


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def exec_gate(batch_root, label):
    """B-exec-* only after a pilot report whose every run passed every check."""
    if not label.startswith('B-exec'):
        return None
    reports = sorted(batch_root.glob('B-pilot_*/data/batch_report.json'))
    if not reports:
        raise ValueError('Run and review B-pilot first (no collected pilot report found)')
    report = read(reports[-1])
    if report['status'] != 'batch_collected_requires_review' or report['executed_runs'] != len(PILOT):
        raise ValueError('Pilot did not complete all runs: ' + str(report['status']))
    for rid, row in report['results'].items():
        checks = (row.get('exit_code') == 0 and row['binding']['status'] == 'passed'
                  and row['reset']['status'] == 'passed'
                  and row['clock']['status'] == 'verified_recorded_events_labels_match'
                  and row['tool_interface']['status'] == 'passed_v2')
        if not checks:
            raise ValueError('Pilot run did not pass every check: ' + rid)
    return str(reports[-1])


S4_OFFSET_MM = (0.02, 0.0)
NOISE_REPEATS = {'N_candidate_000_r1': 'candidate_000', 'N_candidate_000_r2': 'candidate_000',
                 'N_candidate_012_r1': 'candidate_012', 'N_candidate_012_r2': 'candidate_012'}


def noise_runs(plan):
    """L4-02 B1: zero-offset repeats of the frozen candidate_000 / candidate_012 scenes (4 new simulations)."""
    ids = []
    for rid, cid in NOISE_REPEATS.items():
        base = next(r for r in plan['pool32']['candidates'] if r['candidate_id'] == cid)
        if base['group'] != 'frozen_seed0':
            raise ValueError('Noise repeats use frozen-pool scenes only: ' + cid)
        plan['runs'][rid] = {'run_id': rid, 'candidate_id': cid, 'group': 'noise_repeat',
                             'action_mm_mm_deg': base['parameters_mm_mm_deg'], 'tool_offset_mm': [0.0, 0.0],
                             'scene': base['scene'], 'scene_sha256': base['scene_sha256'], 'scene_origin': 'frozen_pool'}
        ids.append(rid)
    return ids


def s4_runs(plan, batch_root, source_pool, requested):
    """S4 = the A7 rule's choice from the latest Phase C report, re-run with a (0.02, 0) mm tool offset."""
    for label in ('B-exec-1', 'B-exec-2', 'B-exec-3', 'B-exec-4'):
        if not sorted(batch_root.glob(label + '_*/data/batch_report.json')):
            raise ValueError('B-S4 needs all B-exec reports first (missing %s)' % label)
    reports = sorted(batch_root.glob('C-analysis_*/data/report/phase_c_report.json'))
    if not reports:
        raise ValueError('Run the Phase C analysis first; S4 candidates come from its pre-registered rule')
    chosen = read(reports[-1])['S4_selection']['chosen']
    if requested and list(requested) != chosen:
        raise ValueError('--s4 %s differs from the rule-based choice %s' % (requested, chosen))
    if not 1 <= len(chosen) <= 2:
        raise ValueError('S4 rule must choose 1-2 candidates, got %s' % chosen)
    ids = []
    for cid in chosen:
        base = next(r for r in plan['pool32']['candidates'] if r['candidate_id'] == cid)
        rid = 'S4_%s_x_0p02' % cid
        plan['runs'][rid] = {'run_id': rid, 'candidate_id': cid, 'group': 'sensitivity_S4',
                             'action_mm_mm_deg': base['parameters_mm_mm_deg'], 'tool_offset_mm': list(S4_OFFSET_MM),
                             'scene': rid + '.yaml', 'scene_sha256': None, 'scene_origin': 'generated_s4',
                             'source_scene': base['scene'], 'source_scene_sha256': base['scene_sha256'],
                             'source_origin': 'frozen_pool' if base['group'] == 'frozen_seed0' else 'repo',
                             'phase_c_report': str(reports[-1])}
        ids.append(rid)
    return ids


def write_s4_scenes(plan, tools, source_pool):
    import yaml
    sys.path.insert(0, str(REPO))
    from level4.tool_offset import apply_tool_offset
    for row in plan['runs'].values():
        if row.get('scene_origin') != 'generated_s4':
            continue
        source = (source_pool if row.pop('source_origin') == 'frozen_pool' else CONFIG / 'scenes') / row.pop('source_scene')
        if sha256(source) != row.pop('source_scene_sha256'):
            raise ValueError('S4 source scene differs from the frozen plan: ' + row['run_id'])
        with source.open(encoding='utf-8') as stream:
            scene = yaml.safe_load(stream)
        shifted = apply_tool_offset(scene, row['tool_offset_mm'], 0.2)
        with (tools / row['scene']).open('x', encoding='utf-8') as stream:
            yaml.safe_dump(shifted, stream, sort_keys=False, allow_unicode=True)


def pbs_script(runner):
    return '\n'.join([
        '#!/bin/bash', 'set -euo pipefail', ': "${PROBE_PARENT:?}"',
        'source /home/users/ntu/mingjie0/venvs/doughnet/bin/activate',
        'export WANDB_MODE=offline', 'export PYTHONUNBUFFERED=1', 'export PYTHONDONTWRITEBYTECODE=1',
        'export OMP_NUM_THREADS=1', 'export OPENBLAS_NUM_THREADS=1',
        'finish_probe() {', '  PROBE_STATUS=$?', '  trap - EXIT', '  set +e',
        '  printf \'%s\\n\' "$PROBE_STATUS" > "$PROBE_PARENT/job_exit_code.txt"',
        '  qstat -f "${PBS_JOBID:-unknown}" > "$PROBE_PARENT/qstat_end.txt" 2>&1',
        '  exit "$PROBE_STATUS"', '}', 'trap finish_probe EXIT',
        'printf \'%s\\n\' "${PBS_JOBID:?}" > "$PROBE_PARENT/job_id.txt"',
        'qstat -f "$PBS_JOBID" > "$PROBE_PARENT/qstat_start.txt" 2>&1 || true',
        'python --version > "$PROBE_PARENT/python.txt" 2>&1',
        'nvidia-smi > "$PROBE_PARENT/nvidia-smi.txt"',
        'python -B -u "$PROBE_PARENT/tools/%s" \\' % runner,
        '  --request "$PROBE_PARENT/tools/request.json" --output "$PROBE_PARENT/data" \\',
        '  > "$PROBE_PARENT/collector.log" 2>&1', 'cat "$PROBE_PARENT/collector.log"', ''])


def prepare(label, project, submit, s4=None, walltime=None):
    plan = load_plan()
    batch_root = project / BATCH_DIRNAME
    batch_root.mkdir(parents=True, exist_ok=True)
    pilot_report = exec_gate(batch_root, label)
    frozen_parent, frozen_request, frozen_manifest = frozen22(project)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')  # unique even for retries within a second
    parent = batch_root / ('%s_%s' % (label, stamp))
    parent.mkdir()
    if label == 'B-dyn':
        runner, walltime = 'predict_level4_batch.py', walltime or '03:00:00'
        tools = build_tools(parent, frozen_parent, frozen_request, frozen_manifest, runner)
        request = base_request(frozen_request, 'level4_batch_prediction_v1', label)
        legacy_log = Path(frozen_request['legacy_source_log'])
        legacy_cfg = legacy_log.parent / 'config.yaml'
        nominal = frozen_parent / 'data'
        for path in (legacy_log, legacy_cfg, nominal / 'evidence_manifest.json'):
            request['required_files'].setdefault(str(path), file_entry(path))
        request.update(new_simulations_max=0, variants=['legacy_nominal', 'v3_nominal'],
                       candidates=plan['predict_candidates'], new_Dyn_rollouts=2 * len(plan['predict_candidates']),
                       chunk_size=65536, sampling_seed=0, v3_nominal={'evidence_dir': str(nominal), 'job': '25837718.pbs101'},
                       legacy_validation={'source_log': str(legacy_log), 'scene_config': str(legacy_cfg),
                                          'max_bbox_mm': 1.0, 'max_centroid_mm': 1.0},
                       execution_truth_read=False)
    else:
        runner, walltime = 'run_level4_batch.py', walltime or '01:00:00'
        source_pool = Path(frozen_request['source_pool'])
        pool_hashes = read(source_pool / 'pool.json')['input_sha256']
        if label == 'B-S4':
            plan['jobs']['B-S4'] = s4_runs(plan, batch_root, source_pool, s4)
        if label == 'B-noise':
            plan['jobs']['B-noise'] = noise_runs(plan)
        if label not in plan['jobs']:
            raise ValueError('Unknown job label: ' + label)
        tools = build_tools(parent, frozen_parent, frozen_request, frozen_manifest, runner)
        if label == 'B-S4':
            write_s4_scenes(plan, tools, source_pool)
        runs = []
        for rid in plan['jobs'][label]:
            row = dict(plan['runs'][rid])
            origin = row.pop('scene_origin')
            if origin == 'generated_s4':
                row['scene_sha256'] = sha256(tools / row['scene'])
                runs.append(row)
                continue
            source = source_pool / row['scene'] if origin == 'frozen_pool' else CONFIG / 'scenes' / row['scene']
            if sha256(source) != row['scene_sha256'] or (origin == 'frozen_pool'
                                                          and pool_hashes[row['scene']] != row['scene_sha256']):
                raise ValueError('Scene hash differs from the frozen plan: ' + rid)
            shutil.copyfile(source, tools / row['scene'])
            runs.append(row)
        request = base_request(frozen_request, 'level4_batch_execution_v1', label)
        request.update(runs=runs, new_simulations_max=len(runs), new_Dyn_rollouts=0,
                       stop_rule={'max_failure_fraction': 0.2, 'stop_on_binding_failure': True},
                       tool_interface_contract='tool_interface_contract_v2',
                       nominal_reference={'evidence_dir': str(frozen_parent / 'data'), 'job': '25837718.pbs101'},
                       pilot_report=pilot_report)
        if label == 'B-noise':
            request.update(task_sheet='L4-02', l4_02_plan_document={
                'path': 'docs/level4_l4_02_plan.md', 'sha256': sha256(REPO / 'docs/level4_l4_02_plan.md')})
    write(tools / 'request.json', request)
    with (tools / 'run_job.pbs').open('w', encoding='utf-8', newline='\n') as stream:  # Path.write_text(newline=) needs 3.10
        stream.write(pbs_script(runner))
    write(tools / 'tool_manifest.json', {'files_sha256': {p.name: sha256(p) for p in sorted(tools.iterdir())}})
    syntax = subprocess.run(['bash', '-n', str(tools / 'run_job.pbs')], capture_output=True, text=True)
    if syntax.returncode:
        raise ValueError('PBS script syntax check failed: ' + syntax.stderr)
    sys.path.insert(0, str(tools))
    from resume_level4_shared_input import check_files
    check_files(request['required_files'])
    if label == 'B-dyn':
        # CPU preflight in a separate process and directory: rebuilds both tool inputs
        # (legacy agreement gate included) before any GPU time is requested.
        check = subprocess.run([sys.executable, '-B', str(tools / runner), '--preflight',
                                '--request', str(tools / 'request.json'), '--output', str(parent / 'preflight/data')],
                               capture_output=True, text=True)
        (parent / 'preflight.log').write_text(check.stdout + check.stderr, encoding='utf-8')
        if check.returncode:
            raise RuntimeError('B-dyn CPU preflight failed; see ' + str(parent / 'preflight.log'))
    command = ['qsub', '-N', 'l4_' + label.replace('-', '_'), '-P', 'personal', '-q', 'normal',
               '-l', QSUB_SELECT, '-l', 'walltime=' + walltime, '-j', 'oe', '-o', str(parent / 'pbs.log'),
               '-v', 'PROBE_PARENT=' + str(parent), str(tools / 'run_job.pbs')]
    write(parent / 'login_verification.json', {'status': 'passed', 'required_files': len(request['required_files']),
          'pbs_bash_syntax': 'passed', 'request_sha256': sha256(tools / 'request.json'),
          'tool_manifest_sha256': sha256(tools / 'tool_manifest.json'), 'qsub_command': command})
    print('PREPARED', parent)
    print('REQUEST_SHA256', sha256(tools / 'request.json'))
    if submit:
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(project))
        (parent / 'qsub.stdout.txt').write_text(result.stdout, encoding='utf-8')
        (parent / 'qsub.stderr.txt').write_text(result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('qsub failed; see ' + str(parent))
        (parent / 'submitted_job_id.txt').write_text(result.stdout.strip() + '\n', encoding='utf-8')
        print('SUBMITTED', result.stdout.strip())
    else:
        print('QSUB', ' '.join(command))
    return parent


def status(project):
    for parent in sorted((project / BATCH_DIRNAME).glob('B-*_*')):
        job = (parent / 'submitted_job_id.txt').read_text(encoding='utf-8').strip() \
            if (parent / 'submitted_job_id.txt').is_file() else 'not_submitted'
        exit_code = (parent / 'job_exit_code.txt').read_text(encoding='utf-8').strip() \
            if (parent / 'job_exit_code.txt').is_file() else '-'
        summary = ''
        for name in ('batch_report.json', 'prediction_report.json'):
            path = parent / 'data' / name
            if path.is_file():
                report = read(path)
                summary = '%s executed=%s failed=%s rollouts=%s' % (report.get('status'), report.get('executed_runs'),
                                                                   report.get('failed_runs'), report.get('rollouts'))
        print('%-40s job=%-18s exit=%-3s %s' % (parent.name, job, exit_code, summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    prep = sub.add_parser('prepare')
    prep.add_argument('label', choices=('B-pilot', 'B-exec-1', 'B-exec-2', 'B-exec-3', 'B-exec-4', 'B-dyn', 'B-S4', 'B-noise'))
    prep.add_argument('--submit', action='store_true')
    prep.add_argument('--s4', nargs='*')
    prep.add_argument('--walltime')
    sub.add_parser('status')
    for p in sub.choices.values():
        p.add_argument('--project', type=Path, default=PROJECT)
    args = parser.parse_args()
    if args.command == 'plan':
        plan = load_plan()
        for label, ids in plan['jobs'].items():
            print('%-9s %2d runs: %s' % (label, len(ids), ' '.join(ids)))
        print('B-dyn     %d candidates x 2 tool inputs = %d rollouts' % (
            len(plan['predict_candidates']), 2 * len(plan['predict_candidates'])))
        print('B-S4      after B-exec review: 1-2 runs (pre-registered rule)')
    elif args.command == 'prepare':
        prepare(args.label, args.project, args.submit, args.s4, args.walltime)
    else:
        status(args.project)


if __name__ == '__main__':
    main()
