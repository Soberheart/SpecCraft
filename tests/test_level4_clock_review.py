import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from level4.clock_review import review_clock_schedule


def fixture(root, extra=True):
    tools = root / 'tools'
    tools.mkdir()
    events = []
    active = counter = 0
    def record(kind, **fields):
        events.append(dict(event=len(events) + 1, kind=kind, active_substeps=active, **fields))
    def step():
        nonlocal active, counter
        before = counter
        active += 10
        counter += 10
        record('step', n_substeps=10, solver_counter_before=before, solver_counter_after=counter)
    def save(frame):
        np.savez(tools / ('frame_%03d.npz' % frame), step=np.asarray(frame * 16),
                 sample_global_substep=np.asarray(counter), current_local_substep=np.asarray(counter % 50),
                 previous_local_substep=np.asarray((counter - 1) % 50), substep_dt=np.asarray(.0002),
                 active_branch_clock_known=np.asarray(True), active_branch_substeps=np.asarray(active),
                 clock_event_count=np.asarray(len(events)))
    record('reset_origin', solver_counter_after=0)
    step()
    save(0)
    for _ in range(16):
        step()
    if extra:
        snapshot = active
        record('get_state', snapshot=1)
        for _ in range(3):
            step()
        before = active
        active, counter = snapshot, 0
        record('set_state', source_snapshot=1, before_active_substeps=before, solver_counter_after=0)
        step()
    save(1)
    path = root / 'events.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in events), encoding='utf-8')
    return path, tools


class ClockReviewTests(unittest.TestCase):
    def test_rollback_and_retained_step_are_distinguished(self):
        with tempfile.TemporaryDirectory() as d:
            result = review_clock_schedule(*fixture(Path(d)), frame_count=2)
            row = result['records'][1]
            self.assertEqual(row['interval_discarded_substeps'], 30)
            self.assertEqual(row['interval_extra_retained_substeps'], 10)
            self.assertAlmostEqual(row['active_branch_elapsed_s'], .034)
            self.assertAlmostEqual(row['since_shared_reset_s'], .036)
            self.assertFalse(result['candidate_pool_released'])

    def test_original_schedule_matches_and_frame0_origin_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            result = review_clock_schedule(*fixture(Path(d), extra=False), frame_count=2)
            self.assertIsNone(result['first_time_mismatch_frame'])
            self.assertEqual(result['records'][0]['active_branch_elapsed_s'], 0)
            self.assertEqual(result['records'][0]['since_shared_reset_s'], .002)

    def test_counter_edit_with_consistent_local_indices_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            event_path, tools = fixture(Path(d))
            path = tools / 'frame_001.npz'
            with np.load(path) as packet:
                values = dict(packet)
            # Preserves both modulo buffer indices, but lies about its event prefix.
            values['sample_global_substep'] += 50
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, 'event prefix'):
                review_clock_schedule(event_path, tools, frame_count=2)

    def test_untracked_restore_cannot_be_certified(self):
        with tempfile.TemporaryDirectory() as d:
            event_path, tools = fixture(Path(d))
            events = [json.loads(line) for line in event_path.read_text(encoding='utf-8').splitlines()]
            next(row for row in events if row['kind'] == 'set_state')['source_snapshot'] = None
            event_path.write_text('\n'.join(json.dumps(row) for row in events), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unknown scene restore source'):
                review_clock_schedule(event_path, tools, frame_count=2)
