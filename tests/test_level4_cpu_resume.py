import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from level4.artifacts import read_json,write_json,sha256
from level4.archive import verify_zip
from scripts.prepare_level4_shared_input import validate_level4_closure
from scripts.resume_level4_shared_input import check_files,collect


class CPUResumeTests(unittest.TestCase):
    def test_deferred_missing_archive_rejected_even_when_checkout_contains_it(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);package=root/'level4';package.mkdir()
            (package/'shared_input.py').write_text('def check():\n    from .archive import safe_name\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'dependency missing: level4.archive'):
                validate_level4_closure(root)
            (package/'archive.py').write_text('def safe_name(x): return x\n', encoding='utf-8')
            validate_level4_closure(root)

    def test_closure_uses_delivered_package_in_isolated_subprocess(self):
        repo=Path(__file__).resolve().parents[1]
        from tests.test_level4_shared_input import initial_fixture
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);package=root/'level4';package.mkdir()
            for name in ('__init__.py','shared_input.py','archive.py','artifacts.py','geometry.py'):
                (package/name).write_bytes((repo/'level4'/name).read_bytes())
            source,meta=initial_fixture(root)
            code="import sys;sys.path.insert(0,sys.argv[1]);from level4.shared_input import checked_initial_input;from level4.artifacts import read_json;checked_initial_input(sys.argv[2],read_json(sys.argv[2]+'/metadata.json'));print('CPU_INPUT_PASSED')"
            command=[sys.executable,'-I','-B','-X','utf8','-c',code,str(root),str(source)]
            result=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('CPU_INPUT_PASSED',result.stdout)
            (package/'archive.py').unlink()  # disposable synthetic fixture only
            result=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("No module named 'level4.archive'",result.stderr)

    def test_changed_saved_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'saved';p.write_bytes(b'original')
            entries={str(p):{'size':p.stat().st_size,'sha256':sha256(p)}}
            check_files(entries);p.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'prerequisite changed'):check_files(entries)

    def test_cpu_failure_retains_failed_status_and_verifiable_package(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);request=root/'request.json';write_json(request,{'kind':'wrong'})
            with patch('sys.stdout',io.StringIO()):self.assertEqual(collect(request,root/'data'),1)
            record=read_json(root/'data/stage_exit.json')
            self.assertEqual(record['exit_code'],1);self.assertEqual(record['GPU_calls'],0)
            self.assertFalse(record['new_simulation_run']);self.assertFalse(record['model_prediction_run'])
            self.assertEqual(verify_zip(root/'cpu_preparation_evidence.zip')['status'],'passed')


if __name__=='__main__':unittest.main()
