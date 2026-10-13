"""Independent reset-diagnosis contracts; never relax the strict gate."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile
import numpy as np
from scripts.audit_level4_reset_failure import initial_difference, audit, digest, configuration_differences


class ResetDiagnosticTests(unittest.TestCase):
    def test_configuration_diff_preserves_physical_and_allowed_output_changes(self):
        before = {'sim': {'seed': 0}, 'render': True, 'log': {'base_dir': 'old'}}
        after = {'sim': {'seed': 1}, 'render': False, 'log': {'base_dir': 'new'}}
        self.assertEqual(configuration_differences(before, after), [
            {'path': 'log.base_dir', 'before': 'old', 'after': 'new'},
            {'path': 'render', 'before': True, 'after': False},
            {'path': 'sim.seed', 'before': 0, 'after': 1}])

    def test_known_translation_and_units(self):
        result = initial_difference([[0, 0, 0], [2, 1, 3]], [[.5, 0, 0], [2.5, 1, 3]], .002, .1)
        self.assertEqual(result['index_paired_max_abs_error_mm'], 1)
        self.assertEqual(result['centroid_shift_mm'], [1, 0, 0])
        self.assertEqual(result['actual_world_xyz_spans_mm'], [4, 2, 6])
        self.assertFalse(result['strict_initial_check_passed'])

    def test_boundary_unchanged_and_permutation_is_diagnostic_only(self):
        self.assertTrue(initial_difference([[0, 0, 0]], [[.5, 0, 0]], .002, 1)['strict_initial_check_passed'])
        self.assertFalse(initial_difference([[0, 0, 0]], [[.5001, 0, 0]], .002, 1)['strict_initial_check_passed'])
        result = initial_difference([[0, 0, 0], [2, 1, 3]], [[2, 1, 3], [0, 0, 0]], .001, .001)
        self.assertEqual(result['lexicographic_max_abs_error_mm'], 0)
        self.assertEqual(result['index_paired_max_abs_error_mm'], 3)
        self.assertFalse(result['strict_initial_check_passed'])

    def test_count_mismatch_and_invalid_input(self):
        result = initial_difference([[0, 0, 0]], [[0, 0, 0], [1, 0, 0]], .001, .001)
        self.assertEqual(result['reason'], 'particle_count_or_shape_mismatch')
        self.assertIsNone(result['index_paired_max_abs_error_mm'])
        for value in ([], [[0, 0]], [[float('nan'), 0, 0]], [[float('inf'), 0, 0]]):
            with self.assertRaises(ValueError): initial_difference(value, [[0, 0, 0]], .001, .001)
        for scale, tolerance in ((0, .001), (float('nan'), .001), (.001, -1), (.001, True)):
            with self.assertRaises(ValueError): initial_difference([[0, 0, 0]], [[0, 0, 0]], scale, tolerance)

    def test_real_archive_retains_failure_raw_logs_and_exact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / 'request/run/reset_1/pool'
            execution = pool / 'executions/candidate_000'
            (execution / 'scenes/0000').mkdir(parents=True)
            source = root / 'source'
            source.mkdir()
            reference = np.array([[0., 0, 0], [1, 2, 3]])
            original = [{'step': i * 16, 't': i * .032, 'frame_idx': i,
                         'obj': {'particles': {'pos': reference.copy(), 'vel': np.zeros((2, 3))}}} for i in range(2)]
            actual = copy.deepcopy(original)
            actual[0]['obj']['particles']['pos'] += [0, .25, 0]
            for path, rows in ((source / 'log.pkl', original), (execution / 'scenes/0000/log.pkl', actual)):
                with path.open('wb') as stream: pickle.dump(rows, stream)
            for path in (source / 'config.yaml', pool / 'source_scene.yaml', pool / 'candidate_000.yaml',
                         execution / 'scenes/0000/config.yaml'):
                path.write_text('sim: {seed: 0}\n', encoding='utf-8')
            (pool / 'runtime_snapshot').mkdir()
            (pool / 'runtime_snapshot/manifest.json').write_text('{}', encoding='utf-8')
            metadata = {'scene_dir': str(source), 'source_log_sha256': digest(source / 'log.pkl'),
                        'scene_config_sha256': digest(source / 'config.yaml')}
            (pool / 'source_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            (pool / 'pool_config.json').write_text('{}', encoding='utf-8')
            np.savez(pool / 'planner_input.npz', initial_particles=reference)
            np.savez(pool / 'grid.npz', unit_to_meter=.001)
            inputs = {name: digest(pool / name) for name in ('source_metadata.json', 'planner_input.npz', 'grid.npz')}
            manifest = {'input_sha256': inputs, 'config': {'initial_reset_max_error_mm': .001},
                        'simulator_seed': 0, 'observation_seed': 17, 'candidates': [{'candidate_id': 'candidate_000'}]}
            (pool / 'pool.json').write_text(json.dumps(manifest), encoding='utf-8')
            (execution / 'exit.json').write_text(json.dumps({'exit_code': 0, 'pool_sha256': digest(pool / 'pool.json')}), encoding='utf-8')
            (execution / 'failure.json').write_text('{"stage":"raw_truth_extraction","error":"independent_initial_reset_mismatch"}', encoding='utf-8')
            (execution / 'generate.log').write_text('Done simulating sequence\n', encoding='utf-8')
            archive = root / 'diagnosis.zip'
            output = root / 'evidence'
            result = audit(pool, output, archive)
            self.assertFalse(result['initial_difference']['strict_initial_check_passed'])
            self.assertAlmostEqual(result['initial_difference']['index_paired_max_abs_error_mm'], .25)
            self.assertEqual(result['execution_frame_count'], 2)
            with zipfile.ZipFile(archive) as bundle:
                hashes = json.loads(bundle.read('evidence_manifest.json'))['files_sha256']
                import hashlib
                for name, expected in hashes.items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), expected)
                self.assertIn('source_log.pkl', hashes)
                self.assertIn('execution_log.pkl', hashes)
                self.assertEqual(json.loads(bundle.read('failure.json'))['error'], 'independent_initial_reset_mismatch')
            prior = archive.read_bytes()
            with self.assertRaises(FileExistsError): audit(pool, root / 'new_evidence', archive)
            self.assertEqual(archive.read_bytes(), prior)
            with self.assertRaises(ValueError): audit(pool, pool / 'bad_output', root / 'bad.zip')
            with self.assertRaises(ValueError): audit(pool, root / 'outside', pool / 'bad.zip')
            changed = pool / 'source_metadata.json'
            changed.write_bytes(changed.read_bytes() + b' ')
            with self.assertRaises(ValueError): audit(pool, root / 'changed', root / 'changed.zip')


if __name__ == '__main__':
    unittest.main()
