"""L4-01 A6: extend the frozen 16-candidate development pool to 32 (+ a D2 group).

CPU only; never edits the source pool.  Rules (task sheet L4-01 A6):
1. If generate_actions(seed=0, count=32)[:16] equals the frozen 16 bit-for-bit,
   extend directly; otherwise keep the frozen 16 and register 16 new candidates
   from seed=1 (generate_actions(1, 17)[1:], i.e. without its nominal row) as a
   second group.
2. A separate, explicitly designed ``d2_edge`` group probes the most asymmetric
   collision-free pinches (see D2_EDGE for the rationale); it is reported apart
   from the random groups and never pooled into D1 statistics by default.
3. Every candidate gets a frame0 finger/material clearance pre-screen.
New scene YAMLs are produced from candidate_000.yaml exactly as
``level4.pipeline`` builds them (to_pos += action_xy/(1000*scale); to_quat =
compose_yaw(nominal, yaw)); the method is validated by regenerating the frozen
candidates 001-015 and requiring parsed-YAML equality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from level4.pipeline import generate_actions  # noqa: E402
from level4.finger_clearance import finger_geometry, initial_clearance  # noqa: E402
from scripts.plan_level3_pose_cem import compose_yaw  # noqa: E402

SOURCE_POOL_SHA256 = '5a62a14376e1d0f9c10565c9915a536e1cc088d41cfa587a3e5fac03df2c990a'
BOUNDS = np.array([40.0, 40.0, 10.0])
NOMINAL_YAW_DEG = 30.0
_C = np.array([np.cos(np.radians(NOMINAL_YAW_DEG)), np.sin(np.radians(NOMINAL_YAW_DEG))])  # closing axis
_D = np.array([-_C[1], _C[0]])  # finger-width axis


def _d2(s, t, yaw):
    xy = s * _C + t * _D
    return [float(xy[0]), float(xy[1]), float(yaw)]


# Designed before any execution; s = offset along the nominal closing axis,
# t = along the finger-width axis (mm).  Fingers are 10 mm wide, start 100 mm
# apart at table level and close to 1 mm, so a cut needs the pinch plane where
# the material chord is <= ~10 mm, which the frame0 clearance forbids (|s| <~ 20).
# These four are the most asymmetric pinches that stay collision-free, to test
# empirically whether any break/no-break boundary is reachable in the contract.
D2_EDGE = {
    'd2_closing_axis_plus': (_d2(18.0, 0.0, 0.0), 'pinch plane shifted +18 mm along closing axis'),
    'd2_closing_axis_minus': (_d2(-18.0, 0.0, 0.0), 'pinch plane shifted -18 mm along closing axis'),
    'd2_width_axis_edge': (_d2(0.0, 28.0, 0.0), 'fingers straddle the material edge along the finger-width axis'),
    'd2_corner_yaw': (_d2(14.0, 20.0, 10.0), 'combined shift toward a corner with +10 deg yaw'),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def scene_for(base, action, scale, base_dir):
    scene = json.loads(json.dumps(base))
    centre = np.asarray(base['actions']['grasp']['to_pos'], dtype=float)
    position = centre.copy()
    position[:2] += np.asarray(action[:2]) / (1000 * scale)
    scene['actions']['grasp']['to_pos'] = position.tolist()
    scene['actions']['grasp']['to_quat'] = compose_yaw(base['actions']['grasp']['to_quat'], action[2])
    scene['log']['base_dir'] = base_dir
    return scene


def build(source, output, server_pool_dir):
    source, output = Path(source).resolve(), Path(output)
    if sha256(source / 'pool.json') != SOURCE_POOL_SHA256:
        raise ValueError('Source pool.json is not the frozen 5a62a143... pool')
    pool = json.loads((source / 'pool.json').read_text(encoding='utf-8'))
    frozen = np.array([row['parameters_mm_mm_deg'] for row in pool['candidates']], dtype=float)
    for row in pool['candidates']:
        if sha256(source / row['scene']) != pool['input_sha256'][row['scene']]:
            raise ValueError('Frozen candidate scene changed: ' + row['scene'])
    scale = float(json.loads((source / 'source_metadata.json').read_text(encoding='utf-8'))['unit_to_meter'])
    seed0_16 = generate_actions(0, 16)
    if not np.array_equal(seed0_16, frozen):
        raise ValueError('generate_actions(0,16) no longer reproduces the frozen pool')
    seed0_32 = generate_actions(0, 32)
    prefix_equal = bool(np.array_equal(seed0_32[:16], frozen))
    differing_rows = [int(i) for i in range(16) if not np.array_equal(seed0_32[i], frozen[i])]
    if prefix_equal:
        new_group, new_actions, rule = 'seed0_extension', seed0_32[16:], 'generate_actions(0,32)[16:]'
    else:
        new_group, new_actions, rule = 'seed1_second_group', generate_actions(1, 17)[1:], 'generate_actions(1,17)[1:]'
    base = read_yaml(source / 'candidate_000.yaml')
    # Validate the YAML construction method on the frozen candidates.
    for row in pool['candidates'][1:]:
        stored = read_yaml(source / row['scene'])
        rebuilt = scene_for(base, row['parameters_mm_mm_deg'], scale, stored['log']['base_dir'])
        if rebuilt != stored:
            raise ValueError('Scene construction does not reproduce frozen ' + row['scene'])
    state = np.load(source / 'shared_source_state.npz', allow_pickle=False)
    particles = state['mpm_pos'].astype(float) * 1000 * scale
    geometry = finger_geometry(read_yaml(source / 'source_scene.yaml'))
    output.mkdir(parents=False, exist_ok=False)
    rows = []
    for index, row in enumerate(pool['candidates']):
        rows.append({'candidate_id': row['candidate_id'], 'group': 'frozen_seed0', 'design': 'frozen',
                     'parameters_mm_mm_deg': row['parameters_mm_mm_deg'], 'scene': row['scene'],
                     'scene_sha256': pool['input_sha256'][row['scene']], 'scene_source': 'frozen pool, unchanged'})
    extra = [(new_group, rule, list(map(float, a))) for a in new_actions] + \
            [('d2_edge', name, action) for name, (action, _) in D2_EDGE.items()]
    for offset, (group, design, action) in enumerate(extra):
        action = np.asarray(action, dtype=float)
        if (np.abs(action) > BOUNDS).any() or not np.isfinite(action).all():
            raise ValueError('Candidate outside development bounds')
        cid = 'candidate_%03d' % (16 + offset)
        scene = scene_for(base, action, scale, '%s/executions/%s/scenes' % (server_pool_dir, cid))
        path = output / (cid + '.yaml')
        with path.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(scene, stream, sort_keys=False, allow_unicode=True)
        if read_yaml(path) != scene:
            raise ValueError('YAML round trip changed ' + cid)
        rows.append({'candidate_id': cid, 'group': group, 'design': design,
                     'parameters_mm_mm_deg': action.tolist(), 'scene': path.name, 'scene_sha256': sha256(path),
                     'scene_source': 'new, built from frozen candidate_000.yaml'})
    all_actions = np.array([r['parameters_mm_mm_deg'] for r in rows])
    if len(np.unique(all_actions, axis=0)) != len(rows):
        raise ValueError('Duplicate candidate actions')
    for r in rows:
        r['action_cost'] = float(np.linalg.norm(np.asarray(r['parameters_mm_mm_deg']) / BOUNDS))
        c = initial_clearance(particles, geometry, r['parameters_mm_mm_deg'])
        r['frame0_clearance'] = {'min_clearance_mm': c['min_clearance_mm'],
                                 'interpenetrating': c['interpenetrating_at_frame0'],
                                 'particles_within_2mm': sum(f['particles_within']['2.0'] for f in c['fingers'].values())}
        # A candidate whose finger starts inside the material is kept in the
        # registry (no re-draw, no seed shopping) but is not executed or scored.
        r['phase_b_plan'] = ('excluded_pre_screen_interpenetrating_at_frame0' if c['interpenetrating_at_frame0']
                             else 'existing_execution_reuse' if r['candidate_id'] in ('candidate_000', 'candidate_001')
                             else 'execute_and_predict')
    manifest = {
        'schema': 'level4_pool32_l4_01_v1', 'source_pool_sha256': SOURCE_POOL_SHA256,
        'source_pool_directory': str(source), 'server_pool_dir_for_new_scenes': server_pool_dir,
        'seed0_count32_prefix_equals_frozen16': prefix_equal, 'seed0_count32_rows_differing': differing_rows,
        'second_group_rule': rule, 'd2_edge_rationale': {k: v[1] for k, v in D2_EDGE.items()},
        'd2_edge_axes': {'closing_axis_xy': _C.tolist(), 'finger_width_axis_xy': _D.tolist()},
        'frozen16_unchanged': {'ids_actions_scene_hashes_verified': True},
        'scene_method_validated_on_frozen_001_015': True,
        'finger_geometry': geometry, 'clearance_state': 'shared_source_state.npz mpm_pos (reset state)',
        'clearance_state_sha256': sha256(source / 'shared_source_state.npz'),
        'candidates': rows, 'candidate_count': len(rows),
        'groups': {g: [r['candidate_id'] for r in rows if r['group'] == g] for g in dict.fromkeys(r['group'] for r in rows)},
        'phase_b_counts': {k: sum(r['phase_b_plan'] == k for r in rows) for k in
                           ('execute_and_predict', 'existing_execution_reuse',
                            'excluded_pre_screen_interpenetrating_at_frame0')},
        'scope': 'development pool; thresholds null; truth never used for selection; not executed',
    }
    with (output / 'pool32_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-pool-dir', required=True)
    args = parser.parse_args()
    m = build(args.source_pool, args.output, args.server_pool_dir)
    print('POOL32_PREPARED candidates=%d prefix_equal=%s groups=%s' % (
        m['candidate_count'], m['seed0_count32_prefix_equals_frozen16'], {k: len(v) for k, v in m['groups'].items()}))


if __name__ == '__main__':
    main()
