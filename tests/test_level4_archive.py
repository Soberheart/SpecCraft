"""Backup failures must be visible without changing scientific outcomes."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from level4.artifacts import write_json, read_json, sha256
from level4.archive import (inventory, copy_archive, verify_files, pack_archive,
                           verify_zip, unpack_archive, stable_hash, record_run, check_pool_records)


class ArchiveTests(unittest.TestCase):
    def fixture(self, root):
        src = root / 'source'; src.mkdir()
        (src / 'failed.log').write_text('simulation failed; never erase this')
        request = {'run': {'run_id': 'failure_01', 'level': 4, 'data_kind': 'independent_simulation',
                          'scientific_status': 'failed', 'stability_basis': 'retained exit record', 'requirements_checked': True},
                   'sources': [{'id': 'run', 'path': str(src)}]}
        manifest = inventory(request)
        path = root / 'manifest.json'; write_json(path, manifest)
        return src, request, manifest, path

    def test_copy_verify_pack_unpack_and_failure_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src, _, manifest, path = self.fixture(root)
            result = copy_archive(path, root / 'backup')
            self.assertEqual(result['scientific_status'], 'failed')
            self.assertEqual(result['statuses']['verified'], 'passed')
            packed = pack_archive(root / 'backup', root / 'backup.zip')
            unpack_archive(root / 'backup.zip', root / 'restore', packed['sha256'], 1024**2)
            verify_files(root / 'restore/files', manifest)
            with self.assertRaises(FileExistsError): copy_archive(path, root / 'backup')
            self.assertEqual((src / 'failed.log').read_text(), 'simulation failed; never erase this')

    def test_missing_dependency_and_hash_only_never_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, req, _, _ = self.fixture(root)
            req['run']['dependencies'] = [{'role': 'weights', 'sha256': '0' * 64}]
            result = inventory(req)
            self.assertEqual(result['statuses']['archive_complete'], 'pending')
            self.assertEqual(result['gaps'][0]['reason'], 'reference_only_no_body_source')
            req['sources'].append({'id': 'weights', 'path': str(root / 'absent.pth')})
            self.assertTrue(any(g['reason'] == 'missing_source' for g in inventory(req)['gaps']))

    def test_running_and_unhashed_cannot_copy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, req, _, path = self.fixture(root)
            for state in ('running', 'unverified'):
                req['run']['scientific_status'] = state
                manifest = inventory(req)
                p = root / (state + '.json'); write_json(p, manifest)
                with self.assertRaises(ValueError): copy_archive(p, root / ('bad_' + state))
            self.assertFalse(inventory(req, False)['files_sha256'])

    def test_tampering_missing_extra_and_truncation_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, _, manifest, path = self.fixture(root)
            copy_archive(path, root / 'backup')
            dest = root / 'backup/files/run/failed.log'
            dest.write_text('truncated')
            with self.assertRaises(ValueError): verify_files(root / 'backup/files', manifest)
            dest.unlink()
            with self.assertRaises(ValueError): verify_files(root / 'backup/files', manifest)

    def test_source_changed_or_added_and_nested_destination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src, _, _, path = self.fixture(root)
            with self.assertRaises(ValueError): copy_archive(path, src / 'backup')
            (src / 'new.log').write_text('new')
            with self.assertRaises(ValueError): copy_archive(path, root / 'backup')

    def test_unstable_hash_and_copy_retains_partial_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src, _, _, path = self.fixture(root)
            with patch('level4.archive.signature', side_effect=[(1, 2), (2, 3)]):
                with self.assertRaises(ValueError): stable_hash(src / 'failed.log')
            with patch('level4.archive.shutil.copyfileobj', side_effect=OSError('transfer failed')):
                with self.assertRaises(OSError): copy_archive(path, root / 'backup')
            failure = read_json(root / 'backup/transfer_failure.json')
            self.assertEqual(failure['scientific_status'], 'failed')
            self.assertTrue(failure['partial_files_retained'])

    def test_unsafe_duplicate_and_unmanifested_zip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, names in enumerate((['../evil'], ['a', 'a'], ['arbitrary_data'])):
                p = root / ('bad%d.zip' % i)
                with zipfile.ZipFile(p, 'x') as z:
                    for name in names: z.writestr(name, b'fake')
                with self.assertRaises(ValueError): verify_zip(p)

    def test_credentials_excluded_and_symlink_is_gap(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src, req, _, _ = self.fixture(root)
            (src / '.env').write_text('private')
            (src / '.ssh').mkdir(); (src / '.ssh/id_rsa').write_text('private')
            result = inventory(req)
            self.assertEqual(len(result['files']), 1)
            self.assertEqual(len(result['exclusions']), 2)
            # Mocked link is portable on Windows without administrator symlink privileges.
            original = Path.is_symlink
            with patch.object(Path, 'is_symlink', lambda p: True if p == src else original(p)):
                result = inventory(req)
            self.assertEqual(result['gaps'][0]['reason'], 'symlink_requires_explicit_resolved_body')

    def test_not_executed_never_filled_from_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); pool = root / 'pool'; pool.mkdir()
            write_json(pool / 'pool.json', {'candidates': [{'candidate_id': 'candidate_000'}]})
            write_json(pool / 'predictions/candidate_000/complete.json', {'status': 'predicted'})
            row = check_pool_records(pool)['candidates'][0]
            self.assertEqual(row['execution_status'], 'not_executed')
            self.assertFalse(row['actual_fields_from_prediction'])

    def evidence_zip(self, root, name, missing=False, tamper=False, extra=False):
        import hashlib
        import json
        inventory_body = json.dumps({'kind': 'experiment_archive_inventory', 'files': {
            'remote/weight.pth': {'source_path': '/nscc/weight.pth', 'size': 400000000}},
            'files_sha256': {}}).encode()
        summary = b'{"mode":"capacity_metadata_only"}'
        checks = {'kind': 'server_capacity_inventory_not_raw_backup', 'files_sha256': {
            'manifest.json': hashlib.sha256(inventory_body).hexdigest(),
            'summary.json': hashlib.sha256(summary).hexdigest()}}
        p = root / name
        with zipfile.ZipFile(p, 'x') as z:
            z.writestr('manifest.json', inventory_body)
            if not missing: z.writestr('summary.json', b'tamper' if tamper else summary)
            z.writestr('evidence_manifest.json', json.dumps(checks))
            if extra: z.writestr('extra', 'unmanifested')
        return p

    def test_inventory_zip_uses_transport_manifest_and_unpacks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = self.evidence_zip(root, 'inventory.zip')
            check = verify_zip(archive)
            self.assertEqual(check['manifest_role'], 'evidence_package_files')
            self.assertEqual(check['files'], 2)
            unpack_archive(archive, root / 'checked', sha256(archive), 1024**2)
            self.assertTrue((root / 'checked/summary.json').is_file())
            self.assertFalse((root / 'checked/remote/weight.pth').exists())

    def test_dual_manifest_tampering_missing_extra_members_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, options in enumerate(({'tamper': True}, {'missing': True}, {'extra': True})):
                archive = self.evidence_zip(root, 'bad%d.zip' % i, **options)
                with self.assertRaises(ValueError): verify_zip(archive)

    def test_postrun_evidence_preserves_failed_exit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / 'run'; run.mkdir()
            (run / 'exit_code.txt').write_text('1')
            result = record_run(run, run / 'pool', 1, root / 'run_evidence')
            self.assertEqual(result['scientific_status'], 'failed')
            self.assertEqual(result['statuses']['archive_complete'], 'pending')


if __name__ == '__main__': unittest.main()
