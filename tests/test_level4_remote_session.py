import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import level4_remote_session as remote


JOB='123.pbs101'
PARENT=remote.SERVER+'/records/level4_shared_reset_'+'a'*32


def framed(rows, bodies):
    header={'job':JOB,'remote_parent':PARENT,'qstat':'job_state = F','files':rows}
    return io.BytesIO(remote.MAGIC+json.dumps(header).encode()+b'\n'+b''.join(bodies))


def row(name, data):
    return {'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


class RemoteSessionTests(unittest.TestCase):
    def test_shell_quote_roundtrip_preserves_data(self):
        data="x'; $(touch injected); `echo injected`\n next"
        self.assertEqual(shlex.split(remote.remote_command(remote.REMOTE_COLLECT,data)),
                         [remote.REMOTE_PYTHON,'-c',remote.REMOTE_COLLECT,data])

    def test_binary_framing_newlines_and_null_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            data=b'\x00\xff\r\nLEVEL4_FILES_V1\n'; rows=[row('pbs.log',data)]
            report=remote.receive_files(framed(rows,[data]),Path(d),JOB,PARENT)
            self.assertEqual((Path(d)/'pbs.log').read_bytes(),data)
            self.assertEqual(report['file_hashes'],'passed')

    def test_truncation_retains_partial_and_never_certifies(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Truncated'):
                remote.receive_files(framed([row('pbs.log',b'1234')],[b'12']),Path(d),JOB,PARENT)
            self.assertEqual((Path(d)/'pbs.log.partial').read_bytes(),b'12')
            self.assertFalse((Path(d)/'pbs.log').exists())

    def test_wrong_hash_never_promotes_partial(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'hash differs'):
                remote.receive_files(framed([row('pbs.log',b'12')],[b'99']),Path(d),JOB,PARENT)
            self.assertFalse((Path(d)/'pbs.log').exists())

    def test_unsafe_duplicate_and_over_budget_rejected_before_writes(self):
        samples=[[row('../escape',b'x')],[row('x',b'x'),row('x',b'x')],
                 [dict(row('x',b''),bytes=2*1024**3+1)],[dict(row('x',b''),bytes=True)]]
        for rows in samples:
            with self.subTest(rows=rows),tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError): remote.receive_files(framed(rows,[]),Path(d),JOB,PARENT)
                self.assertEqual(list(Path(d).iterdir()),[])

    def test_foreign_job_and_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'source differs'):
                remote.receive_files(framed([],[]),Path(d),'999.pbs101',PARENT)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'trailing'):
                remote.receive_files(framed([],[b'junk']),Path(d),JOB,PARENT)

    def test_prior_complete_copy_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'pbs.log').write_bytes(b'old')
            with self.assertRaises(FileExistsError):
                remote.receive_files(framed([row('pbs.log',b'new')],[b'new']),Path(d),JOB,PARENT)
            self.assertEqual((Path(d)/'pbs.log').read_bytes(),b'old')

    def test_remote_finished_state_and_actual_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d); (parent/'pbs.log').write_bytes(b'actual log\r\n')
            # Run the actual remote program with only qstat and allowed-root
            # validation substituted. It cannot contact a server or submit.
            code=remote.REMOTE_COLLECT.replace(
                "r'/home/users/ntu/mingjie0/projects/doughnet/records/level4_shared_reset_[0-9a-f]{32}'",repr(re.escape(str(parent))))
            raw=io.BytesIO(); stdout=SimpleNamespace(buffer=raw)
            result=SimpleNamespace(returncode=0,stdout='Job Id: 123.pbs101\n    job_state = F\n',stderr='')
            with patch.object(sys,'argv',['remote',str(parent),JOB]),patch.object(sys,'stdout',stdout),patch('subprocess.run',return_value=result) as run:
                exec(compile(code,'remote_collect','exec'),{})
            self.assertEqual(run.call_args.args[0],['qstat','-xf',JOB])
            raw.seek(0)
            with tempfile.TemporaryDirectory() as out:
                remote.receive_files(raw,Path(out),JOB,str(parent))
                self.assertEqual((Path(out)/'pbs.log').read_bytes(),b'actual log\r\n')

    def test_running_job_cannot_stream_live_files(self):
        result=SimpleNamespace(returncode=0,stdout='job_state = R\n',stderr='')
        raw=io.BytesIO()
        code=remote.REMOTE_COLLECT.replace(
            "r'/home/users/ntu/mingjie0/projects/doughnet/records/level4_shared_reset_[0-9a-f]{32}'",repr(re.escape(str(Path(PARENT)))))
        with patch.object(sys,'argv',['remote',PARENT,JOB]),patch.object(sys,'stdout',SimpleNamespace(buffer=raw)),patch.object(sys,'stderr',io.StringIO()),patch('subprocess.run',return_value=result):
            with self.assertRaises(SystemExit) as error: exec(compile(code,'remote_collect','exec'),{})
        self.assertEqual(error.exception.code,2); self.assertEqual(raw.getvalue(),b'')

    def test_remote_submit_validates_toolkit_and_records_job_newline(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/'new'; raw=io.BytesIO()
            source=Path(d)/'source.npz'; source.write_bytes(b'test state')
            request={'shared_reset_state':str(source),'required_files':{str(source):{'size':source.stat().st_size,'sha256':remote.digest(source)}}}
            members={'run_shared_reset.pbs':b'#!/bin/bash\n','request.json':json.dumps(request).encode()}
            manifest={'files_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in members.items()}}
            with zipfile.ZipFile(raw,'w') as z:
                for name,data in members.items(): z.writestr(name,data)
                z.writestr('tool_manifest.json',json.dumps(manifest))
            data=raw.getvalue(); stdout=io.StringIO()
            code=remote.REMOTE_SUBMIT.replace(
                "r'/home/users/ntu/mingjie0/projects/doughnet/records/level4_shared_reset_[0-9a-f]{32}'",repr(re.escape(str(parent))))
            def command_reply(command,**kwargs):
                return SimpleNamespace(returncode=0,stdout=JOB+'\n' if command[0]=='qsub' else '',stderr='')
            with patch.object(sys,'argv',['remote',str(parent),hashlib.sha256(data).hexdigest(),str(len(data))]),patch.object(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(data))),patch.object(sys,'stdout',stdout),patch('subprocess.run',side_effect=command_reply) as run:
                exec(compile(code,'remote_submit','exec'),{})
            self.assertEqual(run.call_count,2)
            self.assertEqual((parent/'submitted_job_id.txt').read_text(encoding='utf-8'),JOB+'\n')
            self.assertTrue(stdout.getvalue().startswith('LEVEL4_SUBMISSION_JSON='))
            self.assertEqual((parent/'tools/run_shared_reset.pbs').read_bytes(),members['run_shared_reset.pbs'])

    def test_collect_uses_one_connection_then_checks_bundle_contents(self):
        import zipfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); base=root/'approved'; batch=base/'batch'; batch.mkdir(parents=True)
            remote.write_json(batch/'submission.json',{'job':JOB,'remote_parent':PARENT})
            archive=io.BytesIO(); payload=b'{"kind":"synthetic_protocol_test"}'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('report.json',payload)
                z.writestr('evidence_manifest.json',json.dumps({'files_sha256':{'report.json':hashlib.sha256(payload).hexdigest()}}))
            data=archive.getvalue()
            process=SimpleNamespace(stdout=framed([row('diagnostic_evidence.zip',data),row('job_exit_code.txt',b'0\n')],[data,b'0\n']),wait=lambda:0,poll=lambda:0)
            with patch.object(remote,'APPROVED_DESTINATION',base),patch('subprocess.Popen',return_value=process) as start:
                remote.collect(batch)
            self.assertEqual(start.call_count,1)
            collection=next(batch.glob('collection_*'))
            self.assertEqual((collection/'evidence/report.json').read_bytes(),payload)
            self.assertEqual(json.loads((collection/'internal_verification.json').read_text(encoding='utf-8'))['internal_verification']['status'],'passed')

    def test_submit_opens_one_connection_and_saves_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); base=root/'approved'; base.mkdir(); archive=root/'tools.zip'; archive.write_bytes(b'fake toolkit')
            delivery=root/'delivery.json'; delivery.write_text(json.dumps({'archive_name':'tools.zip','archive_bytes':archive.stat().st_size,'archive_sha256':remote.digest(archive)}), encoding='utf-8')
            def ssh_reply(command,**kwargs):
                self.assertEqual(command[:3],['ssh','-T','NSCC-ASPIRE2A'])
                self.assertEqual(kwargs['input'],b'fake toolkit')
                parent=shlex.split(command[3])[3]
                return SimpleNamespace(returncode=0,stdout=('LEVEL4_SUBMISSION_JSON='+json.dumps({'remote_parent':parent,'job':JOB})+'\n').encode())
            with patch.object(remote,'APPROVED_DESTINATION',base),patch('subprocess.run',side_effect=ssh_reply) as run:
                remote.submit(delivery)
            self.assertEqual(run.call_count,1)
            receipt=json.loads(next(base.glob('*/submission.json')).read_text(encoding='utf-8'))
            self.assertEqual(receipt['ssh_connections'],1)

    def test_failed_submission_retains_logs_and_no_confirmed_job(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); base=root/'approved'; base.mkdir(); archive=root/'tools.zip'; archive.write_bytes(b'x')
            delivery=root/'delivery.json'; delivery.write_text(json.dumps({'archive_name':'tools.zip','archive_bytes':1,'archive_sha256':remote.digest(archive)}), encoding='utf-8')
            with patch.object(remote,'APPROVED_DESTINATION',base),patch('subprocess.run',return_value=SimpleNamespace(returncode=1,stdout=b'failed')):
                with self.assertRaises(RuntimeError): remote.submit(delivery)
            self.assertEqual(len(list(base.glob('*/submission_failure.json'))),1)
            self.assertEqual(list(base.glob('*/submission.json')),[])


if __name__=='__main__': unittest.main()
