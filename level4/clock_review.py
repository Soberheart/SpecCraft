"""Offline clock evidence checks; no simulator, model, or action selection.

Keep the frozen event reducer's report intact and add cross-checks binding
saved solver counters to the exact event prefixes that produced each frame.
"""
import json
from pathlib import Path

import numpy as np

from .candidate_interface import reduce_clock_events


def review_clock_schedule(event_path, tool_directory, frame_count=63):
    replay = reduce_clock_events(event_path, tool_directory, frame_count)
    events = [json.loads(line) for line in Path(event_path).read_text(encoding='utf-8').splitlines()]
    counter = total = restores = 0
    prefixes = {0: (0, 0, 0)}
    for event in events:
        if event['kind'] in ('reset_origin', 'set_state'):
            counter = 0
            restores += event['kind'] == 'set_state'
        elif event['kind'] == 'step':
            counter += event['n_substeps']
            total += event['n_substeps']
        prefixes[event['event']] = (counter, total, restores)
    rows = []
    previous_prefix = -1
    previous = None
    for frame in replay['records']:
        prefix = frame['event_prefix']
        if prefix <= previous_prefix:
            raise ValueError('Frame event prefixes must strictly increase')
        counter, total, restores = prefixes[prefix]
        with np.load(Path(tool_directory) / ('frame_%03d.npz' % frame['frame']), allow_pickle=False) as packet:
            if int(packet['sample_global_substep']) != counter:
                raise ValueError('Saved solver counter differs from its event prefix')
        interval = 0 if previous is None else frame['active_branch_substeps'] - previous['active_branch_substeps']
        executed = 0 if previous is None else total - previous['total_observed_substeps']
        restore_calls = 0 if previous is None else restores - previous['total_restore_calls']
        expected = 0 if previous is None else 160
        rows.append(dict(frame, solver_counter_from_events=counter,
                         total_observed_substeps=total, total_restore_calls=restores,
                         interval_retained_substeps=interval,
                         interval_nominal_substeps=expected,
                         interval_extra_retained_substeps=interval - expected,
                         interval_discarded_substeps=executed - interval,
                         interval_restore_calls=restore_calls,
                         since_shared_reset_s=frame['active_branch_substeps'] * .0002))
        previous_prefix, previous = prefix, rows[-1]
    shifted = [row['frame'] for row in rows if abs(row['elapsed_minus_label_s']) > 1e-8]
    return {'kind': 'offline_clock_prefix_and_schedule_review_v1',
            'status': 'verified_recorded_events_protocol_time_mismatch' if shifted else 'verified_recorded_events_labels_match',
            'frozen_replay': replay, 'records': rows,
            'solver_counter_and_event_prefix_match_all_frames': True,
            'first_time_mismatch_frame': shifted[0] if shifted else None,
            'extra_retained_scene_steps_through_last_saved_frame': sum(row['interval_extra_retained_substeps'] for row in rows) // 10,
            'main_frame60': rows[60] if frame_count > 60 else None,
            'auxiliary_frame62': rows[62] if frame_count > 62 else None,
            'time_origin': 'elapsed_s relative to saved frame0; since_shared_reset_s separately reported',
            'nominal_reference_physical_clock': 'not_recorded_not_certified',
            'historical_time_labels_changed': False, 'task_satisfaction': 'not_evaluated',
            'candidate_pool_released': False, 'new_simulations': 0}
