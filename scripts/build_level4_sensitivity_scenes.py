"""L4-01 A7: S0/S1/S2 (and later S4) execution scenes with a whole-tool offset.

The registered action stays the candidate's own action (prediction input);
only the executed tool trajectory is shifted (level4.tool_offset).  S4 scenes
are built after B1 with ``--s4 candidate_XXX`` from the pool32 directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.tool_offset import apply_tool_offset  # noqa: E402

PLAN = (('S0_repeat_nominal', (0.0, 0.0), 'simulator noise floor vs existing candidate_000'),
        ('S1_x_0p02', (0.02, 0.0), 'tolerance-sized x offset'),
        ('S1_y_0p02', (0.0, 0.02), 'tolerance-sized y offset'),
        ('S2_x_0p1', (0.1, 0.0), 'robot-repeatability-scale reference'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-scene', type=Path, required=True, help='frozen pool candidate_000.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-dir', required=True)
    parser.add_argument('--unit-to-meter', type=float, default=0.2)
    args = parser.parse_args()
    with args.nominal_scene.open(encoding='utf-8') as stream:
        base = yaml.safe_load(stream)
    args.output.mkdir(exist_ok=False)
    rows = []
    for run_id, delta, purpose in PLAN:
        scene = apply_tool_offset(base, delta, args.unit_to_meter)
        scene['log']['base_dir'] = '%s/executions/%s/scenes' % (args.server_dir, run_id)
        path = args.output / (run_id + '.yaml')
        with path.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(scene, stream, sort_keys=False, allow_unicode=True)
        with path.open(encoding='utf-8') as stream:
            if yaml.safe_load(stream) != scene:
                raise ValueError('YAML round trip changed ' + run_id)
        rows.append({'run_id': run_id, 'source_candidate': 'candidate_000', 'action_mm_mm_deg': [0.0, 0.0, 0.0],
                     'tool_offset_mm': list(delta), 'purpose': purpose, 'scene': path.name,
                     'scene_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'prediction_input': 'unchanged nominal action (offset is an unmodelled interface error)'})
    manifest = {'schema': 'level4_sensitivity_scenes_l4_01_v1', 'decision': 'D-1 section 3 (S0-S4)',
                'nominal_scene_sha256': hashlib.sha256(args.nominal_scene.read_bytes()).hexdigest(),
                'runs': rows,
                's4_rule_preregistered': 'after B1: the 1-2 executed candidates with the smallest raw-particle neck '
                                         'width at frame60 (or, if any break, those nearest the break/no-break '
                                         'boundary), each re-run with tool offset (0.02, 0) mm',
                's3b': 'not supported by current single-grasp entries; moved to L5',
                'pass_criteria_preregistered': 'D-1: span and neck change < 0.1 mm and component count unchanged '
                                               '(until epsilon is frozen; then < epsilon/10)'}
    with (args.output / 'sensitivity_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('SENSITIVITY_SCENES_PREPARED runs=%d' % len(rows))


if __name__ == '__main__':
    main()
