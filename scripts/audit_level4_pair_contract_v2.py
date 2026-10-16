"""L4-01 A4: re-judge the saved fixed pair under tool_interface_contract_v2.

CPU only.  Reads the verified job-25837718 evidence; never runs the simulator,
Torch, Taichi or the model and never edits the v1 review.  Steps:
1. run the frozen v1 audit script (hash-checked) and require that it reproduces
   the stored v1 review except for the relocated evidence path;
2. recompute the tool diagnosis from raw saved traces with the current
   ``diagnose_tool_schedule`` and require exact equality with the stored one;
3. judge the stored metrics under v1 (must reproduce ``failed``) and v2.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.tool_interface_contract import evaluate  # noqa: E402

FROZEN_V1_AUDIT_SHA256 = '741d403a7511a1f619d6e208dcb673a248ecd279be60cc2656241c2662cf4f5a'
EXPECTED_JOB = '25837718.pbs101'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_v1(path):
    if sha256(path) != FROZEN_V1_AUDIT_SHA256:
        raise ValueError('v1 audit script differs from the frozen job-25837718 audit')
    spec = importlib.util.spec_from_file_location('frozen_audit_pair_results_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grasp_center(scene_yaml):
    import yaml
    with Path(scene_yaml).open(encoding='utf-8') as stream:
        scene = yaml.safe_load(stream)
    center = [float(value) for value in scene['actions']['grasp']['to_pos']]
    if len(center) != 3:
        raise ValueError('Invalid grasp center')
    return center


def recompute_diagnosis(evidence):
    """Same inputs as probe_level4_transaction.run used on the server."""
    from level4.candidate_interface import diagnose_tool_schedule
    pool = read(evidence / 'source_pool_manifest.json')
    candidate = next(row for row in pool['candidates'] if row['candidate_id'] == 'candidate_001')
    nominal = evidence / 'execution' / 'candidate_000'
    logs = list((nominal / 'scenes').rglob('log.pkl'))
    if len(logs) != 1:
        raise ValueError('Unique prospective nominal log missing')
    return diagnose_tool_schedule(evidence / 'execution' / 'candidate_001', logs[0], nominal / 'actual_tools',
                                  grasp_center(evidence / 'source_scene.yaml'),
                                  candidate['parameters_mm_mm_deg'])


def audit_v2(evidence, v1_script, v1_review):
    evidence = Path(evidence).resolve()
    v1_module = load_v1(v1_script)
    reproduced = v1_module.audit(evidence)  # verifies all 314 evidence hashes
    stored = read(v1_review)
    if {k: v for k, v in reproduced.items() if k != 'evidence_directory'} != \
            {k: v for k, v in stored.items() if k != 'evidence_directory'}:
        raise ValueError('Frozen v1 audit does not reproduce the stored v1 review')
    if stored['job'] != EXPECTED_JOB or stored['gate']['tool_transform_interface'] != 'failed':
        raise ValueError('Stored v1 review is not the expected failed job-25837718 review')
    saved = read(evidence / 'prospective_tool_diagnosis.json')
    fresh = recompute_diagnosis(evidence)
    if fresh['metrics'] != saved['metrics']:
        raise ValueError('Recomputed tool diagnosis differs from the saved diagnosis')
    v1 = evaluate(saved['metrics'], 'tool_interface_contract_v1')
    if v1['status'] != 'failed_v1' or saved['tool_transform_status'] != 'violated':
        raise ValueError('v1 contract does not reproduce the historical violated status')
    v2 = evaluate(saved['metrics'], 'tool_interface_contract_v2')
    gate = dict(stored['gate'])
    gate.update({'tool_transform_interface': v2['status'],
                 'tool_transform_interface_v1_unchanged': stored['gate']['tool_transform_interface'],
                 'model_alignment': 'unverified', 'elapsed_physical_time': 'unknown',
                 'task_constraints': 'not_evaluated'})
    return {
        'schema': 'level4_fixed_pair_audit_v2_contract',
        'decision': 'D-1 (2026-10-16, user-approved; pre-registered before D1/D2/D4 and S0-S4 results)',
        'job': stored['job'], 'evidence_directory': str(evidence),
        'inputs': {'v1_audit_script_sha256': FROZEN_V1_AUDIT_SHA256, 'v1_review_sha256': sha256(v1_review),
                   'prospective_tool_diagnosis_sha256': sha256(evidence / 'prospective_tool_diagnosis.json'),
                   'evidence_manifest_sha256': sha256(evidence / 'evidence_manifest.json')},
        'checks': {'v1_audit_reproduced_except_evidence_directory': True,
                   'evidence_files_verified': reproduced['evidence_files'],
                   'diagnosis_recomputed_from_raw_traces_equal': True,
                   'v1_contract_reproduces_failed': True},
        'v1_result_unchanged': {'tool_transform_interface': 'failed', 'review_file_not_modified': True},
        'contract_v1': v1, 'contract_v2': v2, 'gate': gate,
        'primary_frame_unchanged': 60, 'primary_label_s_unchanged': 1.92,
        'shared_reset_limit_mm_unchanged': 0.001, 'task_thresholds': 'null (uncalibrated)',
        'new_simulations': 0, 'new_Dyn_rollouts': 0, 'candidate_pool_released': False,
        'release_scope': 'tool-interface gate only; Phase B execution requires L4-01 Phase A completion and a frozen request',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--v1-audit-script', type=Path, required=True)
    parser.add_argument('--v1-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_v2(args.evidence, args.v1_audit_script, args.v1_review)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('FIXED_PAIR_AUDIT_V2=' + result['contract_v2']['status'] + '; V1_UNCHANGED=failed; MODEL_ALIGNMENT=unverified')


if __name__ == '__main__':
    main()
