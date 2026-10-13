"""CPU fixtures for source archiving; no physical run or model execution."""
from pathlib import Path
import tempfile
import unittest
from level4.artifacts import write_json, read_json, sha256
from level4.archive import verify_files, unpack_archive
from scripts.backup_level4_sources import backup_sources


class CoreBackupTests(unittest.TestCase):
    def fixture(self, root):
        src = root / 'source'; src.mkdir()
        (src / 'raw.pkl').write_bytes(b'fixture retained failed raw state')
        write_json(src / 'exit.json', {'exit_code': 1})
        weights = root / 'checkpoint.pth'; weights.write_bytes(b'fixture weights')
        return {'run': {'run_id': 'synthetic_backup_fixture', 'level': 4, 'data_kind': 'synthetic',
                       'scientific_status': 'failed', 'requirements_checked': False,
                       'source_science_states': {'123.pbs': {'job_state': 'F', 'exit_code': 1}}},
                'sources': [{'id': 'run', 'path': str(src)}, {'id': 'checkpoint', 'path': str(weights), 'expected_sha256': sha256(weights)}],
                'copy_budget': {'destination_confirmation_required': False, 'maximum_uncompressed_bytes': 1024**2}}

    def test_small_first_then_raw_and_checkpoint_archive_and_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); req = self.fixture(root)
            receipt = backup_sources(req, root / 'backup', scheduler=lambda job: 'job_state = F\nExit_status = 1\n')
            self.assertEqual(receipt['scientific_status'], 'failed')
            self.assertFalse(receipt['simulation_run'])
            self.assertEqual([b['id'] for b in receipt['bundles']], ['small_evidence', 'core_materials'])
            small = read_json(root / 'backup/small_backup/manifest.json')
            self.assertFalse(any(name.endswith(('.pkl', '.pth')) for name in small['files']))
            core = receipt['bundles'][1]
            unpack_archive(core['archive'], root / 'restore', core['sha256'], core['max_uncompressed_bytes'])
            restored = read_json(root / 'restore/manifest.json')
            verify_files(root / 'restore/files', restored)
            self.assertEqual((root / 'restore/files/checkpoint/checkpoint.pth').read_bytes(), b'fixture weights')
            with self.assertRaises(FileExistsError): backup_sources(req, root / 'backup')

    def test_active_job_rejected_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); req = self.fixture(root); digest = sha256(root / 'checkpoint.pth')
            with self.assertRaises(ValueError): backup_sources(req, root / 'backup', scheduler=lambda job: 'job_state = R\n')
            self.assertTrue((root / 'backup/archive_failure.json').is_file())
            self.assertEqual(sha256(root / 'checkpoint.pth'), digest)

    def test_capacity_and_bad_checkpoint_and_destination_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); req = self.fixture(root)
            req['copy_budget']['maximum_uncompressed_bytes'] = 1
            with self.assertRaises(ValueError): backup_sources(req, root / 'too_big', scheduler=lambda job: 'job_state = F\nExit_status = 1\n')
            req['copy_budget']['maximum_uncompressed_bytes'] = 1024**2
            req['sources'][1]['expected_sha256'] = '0' * 64
            with self.assertRaises(ValueError): backup_sources(req, root / 'bad_hash', scheduler=lambda job: 'job_state = F\nExit_status = 1\n')
            req['copy_budget']['destination_confirmation_required'] = True
            with self.assertRaises(ValueError): backup_sources(req, root / 'no_approval')
            self.assertFalse((root / 'no_approval').exists())


if __name__ == '__main__': unittest.main()
