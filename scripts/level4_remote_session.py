"""Submit or collect a frozen Level4 job over one SSH connection per operation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone

SERVER = '/home/users/ntu/mingjie0/projects/doughnet'
REMOTE_PYTHON = '/home/users/ntu/mingjie0/venvs/doughnet/bin/python'
APPROVED_DESTINATION = Path('D:/SpecCraft/level 4')
MAGIC = b'LEVEL4_FILES_V1\n'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda: inp.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def remote_command(code, *args):
    # No interpolation of remote code/data into executable shell expressions.
    return ' '.join(shlex.quote(str(x)) for x in (REMOTE_PYTHON, '-c', code, *args))


REMOTE_SUBMIT = r'''
import hashlib,io,json,os,re,subprocess,sys,zipfile
from pathlib import Path
parent=Path(sys.argv[1]); expected=sys.argv[2]; length=int(sys.argv[3])
if not re.fullmatch(r'/home/users/ntu/mingjie0/projects/doughnet/records/level4_shared_reset_[0-9a-f]{32}',str(parent)):
    raise ValueError('Unexpected output directory')
if length<1 or length>8*1024**2: raise ValueError('Toolkit size exceeds limit')
parent.mkdir(exist_ok=False)
data=sys.stdin.buffer.read(length+1)
with (parent/'toolkit.zip').open('xb') as out: out.write(data)
if len(data)!=length or hashlib.sha256(data).hexdigest()!=expected:
    raise ValueError('Toolkit transport hash/size differs; partial retained')
tools=parent/'tools'; tools.mkdir()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    names=z.namelist()
    if len(names)!=len(set(names)): raise ValueError('Duplicate ZIP members')
    manifest=json.loads(z.read('tool_manifest.json'))['files_sha256']
    if set(names)!=set(manifest)|{'tool_manifest.json'}: raise ValueError('Unexpected toolkit file set')
    if sum(i.file_size for i in z.infolist())>8*1024**2: raise ValueError('Expanded toolkit exceeds limit')
    for name,h in manifest.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or hashlib.sha256(z.read(name)).hexdigest()!=h:
            raise ValueError('Toolkit file hash/path differs: '+name)
    for name in names:
        with (tools/name).open('xb') as out: out.write(z.read(name))
syntax=subprocess.run(['bash','-n',str(tools/'run_shared_reset.pbs')],capture_output=True,text=True)
(parent/'bash_syntax.txt').write_text(syntax.stdout+syntax.stderr)
if syntax.returncode: raise ValueError('PBS syntax failed before submission')
request=json.loads((tools/'request.json').read_text())
if not request.get('shared_reset_state') or request['shared_reset_state'] not in request['required_files']:
    raise ValueError('Frozen shared source body is not registered')
for name,record in request['required_files'].items():
    source=Path(name)
    if source.is_symlink() or source.stat().st_size!=record['size']:
        raise ValueError('Frozen input size/path differs before qsub: '+name)
    h=hashlib.sha256()
    with source.open('rb') as inp:
        for block in iter(lambda:inp.read(1024**2),b''): h.update(block)
    if h.hexdigest()!=record['sha256']: raise ValueError('Frozen input hash differs before qsub: '+name)
with (parent/'login_input_verification.json').open('x') as out:
    json.dump({'status':'passed','files':len(request['required_files']),'before_qsub':True},out,indent=2)
command=['qsub','-N','doughnet_l4_shared_reset','-P','personal','-q','normal','-l',
         'select=1:ngpus=1:ncpus=8:mem=64GB','-l','walltime=00:30:00','-j','oe','-o',str(parent/'pbs.log'),
         '-v','PROBE_PARENT='+str(parent),str(tools/'run_shared_reset.pbs')]
with (parent/'submission_command.json').open('x') as out: json.dump(command,out,indent=2)
result=subprocess.run(command,capture_output=True,text=True,cwd='/home/users/ntu/mingjie0/projects/doughnet')
(parent/'qsub.stdout.txt').write_text(result.stdout); (parent/'qsub.stderr.txt').write_text(result.stderr)
if result.returncode: raise RuntimeError('qsub failed; request evidence retained at '+str(parent))
job=result.stdout.strip()
if not re.fullmatch(r'[0-9]+[.]pbs[0-9]+',job): raise ValueError('Unexpected qsub result; inspect '+str(parent))
(parent/'submitted_job_id.txt').write_text(job+'\n')
print('LEVEL4_SUBMISSION_JSON='+json.dumps({'job':job,'remote_parent':str(parent),
       'diagnostic_only':True,'reset_gate_released':False,'toolkit_sha256':expected}))
'''


REMOTE_COLLECT = r'''
import hashlib,json,re,subprocess,sys
from pathlib import Path
parent=Path(sys.argv[1]); job=sys.argv[2]
if not re.fullmatch(r'/home/users/ntu/mingjie0/projects/doughnet/records/level4_shared_reset_[0-9a-f]{32}',str(parent)):
    raise ValueError('Unexpected collection directory')
if not re.fullmatch(r'[0-9]+[.]pbs[0-9]+',job): raise ValueError('Unexpected job ID')
status=subprocess.run(['qstat','-xf',job],capture_output=True,text=True)
state=re.search(r'job_state\s*=\s*([A-Z])',status.stdout)
if status.returncode or not state or state.group(1)!='F':
    print('Job is not F; no running files copied.\n'+status.stdout+status.stderr,file=sys.stderr)
    sys.exit(2)
items=[]
for name in ('pbs.log','collector.log','job_exit_code.txt','qstat_start.txt','qstat_end.txt',
             'nvidia-smi.txt','python.txt','diagnostic_evidence.zip','tools/request.json','tools/tool_manifest.json',
             'login_input_verification.json','bash_syntax.txt','submission_command.json','submitted_job_id.txt',
             'qsub.stdout.txt','qsub.stderr.txt'):
    path=parent/name
    if not path.is_file(): continue
    if path.is_symlink(): raise ValueError('Unexpected symlink: '+name)
    h=hashlib.sha256()
    with path.open('rb') as inp:
        for block in iter(lambda:inp.read(1024**2),b''): h.update(block)
    items.append({'name':name.replace('tools/','tool_'),'path':path,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
if sum(i['bytes'] for i in items)>2*1024**3: raise ValueError('Collection exceeds 2 GiB budget')
qstat=(status.stdout+status.stderr).encode()
header={'job':job,'remote_parent':str(parent),'qstat':qstat.decode(),
        'files':[{k:i[k] for k in ('name','bytes','sha256')} for i in items]}
out=sys.stdout.buffer
out.write(b'LEVEL4_FILES_V1\n'); out.write(json.dumps(header).encode()+b'\n'); out.flush()
for item in items:
    h=hashlib.sha256(); count=0
    with item['path'].open('rb') as inp:
        for block in iter(lambda:inp.read(1024**2),b''):
            out.write(block); h.update(block); count+=len(block)
    out.flush()
    if count!=item['bytes'] or h.hexdigest()!=item['sha256']:
        raise ValueError('Source changed during stream; reject local copy')
'''


def receive_files(stream, destination, job, parent):
    """Size-bounded binary framing; failed transfers stay as partial files."""
    banner = []
    for _ in range(64):
        line = stream.readline(1024)
        if line == MAGIC:
            break
        if not line:
            raise ValueError('Remote collection stopped before the file header')
        banner.append(line)
    else:
        raise ValueError('Unexpected remote stdout; header not found')
    raw_header = stream.readline(1024**2)
    if not raw_header.endswith(b'\n'):
        raise ValueError('Missing/budget-exceeding transfer header')
    header = json.loads(raw_header)
    if header['job'] != job or header['remote_parent'] != parent:
        raise ValueError('Transfer source differs from saved submission')
    files = header['files']
    if len(files)>20 or len({r['name'] for r in files})!=len(files):
        raise ValueError('Unexpected or duplicate transfer file set')
    total = 0
    for row in files:
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+',row['name']) or row['name'].startswith('.')
                or isinstance(row['bytes'], bool) or not isinstance(row['bytes'], int) or row['bytes']<0
                or not re.fullmatch(r'[0-9a-f]{64}',row['sha256'])):
            raise ValueError('Invalid transfer member')
        total += row['bytes']
    if total>2*1024**3:
        raise ValueError('Transfer exceeds explicit 2 GiB limit')
    if shutil.disk_usage(destination).free<2*total+16*1024**2:
        raise ValueError('Insufficient local disk for transfer')
    write_json(destination/'transfer_header.json', header)
    if banner:
        with (destination/'remote_banner.txt').open('xb') as out:
            out.write(b''.join(banner))
    for row in files:
        part = destination/(row['name']+'.partial')
        h = hashlib.sha256(); left=row['bytes']
        with part.open('xb') as out:
            while left:
                block=stream.read(min(left,1024**2))
                if not block:
                    raise ValueError('Truncated transfer; partial retained: '+row['name'])
                out.write(block); h.update(block); left-=len(block)
        if h.hexdigest()!=row['sha256']:
            raise ValueError('Transferred file hash differs; partial retained: '+row['name'])
        final=destination/row['name']
        # Exclusive final write preserves old attempts on all OSes.
        with part.open('rb') as inp, final.open('xb') as out:
            shutil.copyfileobj(inp,out)
    if stream.read(1):
        raise ValueError('Unexpected trailing transfer bytes')
    with (destination/'qstat_finished.txt').open('x',encoding='utf-8') as out:
        out.write(header['qstat'])
    return {'files':len(files),'bytes':total,'file_hashes':'passed','scientific_exit_not_reclassified':True}


def approved_directory(path):
    path=Path(path).resolve()
    base=APPROVED_DESTINATION.resolve()
    if path==base or base not in path.parents:
        raise ValueError('Directory must be a batch under the approved D destination')
    return path


def archive_module_root():
    # Frozen copies live in records/<batch>; resolve independently of cwd.
    for parent in Path(__file__).resolve().parents:
        if (parent/'level4/archive.py').is_file():
            return parent
    raise RuntimeError('Cannot locate saved Level4 archive implementation')


def submit(delivery_path):
    delivery=json.loads(delivery_path.read_text(encoding='utf-8'))
    archive=delivery_path.parent/delivery['archive_name']
    if archive.stat().st_size!=delivery['archive_bytes'] or digest(archive)!=delivery['archive_sha256']:
        raise ValueError('Frozen toolkit differs; not submitted')
    if shutil.disk_usage(APPROVED_DESTINATION).free<4*1024**3:
        raise ValueError('At least 4 GiB D free required')
    token=uuid.uuid4().hex
    batch=APPROVED_DESTINATION/('shared_reset_20261012_'+token)
    batch.mkdir(exist_ok=False)
    parent=SERVER+'/records/level4_shared_reset_'+token
    command=['ssh','-T','NSCC-ASPIRE2A',remote_command(REMOTE_SUBMIT,parent,delivery['archive_sha256'],archive.stat().st_size)]
    write_json(batch/'command.json',{'command':command,'created_utc':utc_now(),'connections':1})
    print('输入一次 NSCC 密码，上传与提交将在同一个 SSH 连接完成。',flush=True)
    process=subprocess.run(command,input=archive.read_bytes(),stdout=subprocess.PIPE)
    with (batch/'submission_stdout.txt').open('xb') as out:
        out.write(process.stdout)
    lines=process.stdout.decode('utf-8',errors='replace').splitlines()
    receipts=[line.split('=',1)[1] for line in lines if line.startswith('LEVEL4_SUBMISSION_JSON=')]
    if process.returncode or len(receipts)!=1:
        write_json(batch/'submission_failure.json',{'created_utc':utc_now(),'ssh_exit_code':process.returncode,
            'remote_parent':parent,'submission_unconfirmed':True,'do_not_blindly_resubmit':True})
        raise RuntimeError('Submission unconfirmed; inspect '+str(batch)+'/submission_stdout.txt; do not blindly resubmit')
    receipt=json.loads(receipts[0])
    if receipt['remote_parent']!=parent or not re.fullmatch(r'\d+[.]pbs\d+',receipt['job']):
        raise ValueError('Submission receipt source/job differs')
    receipt.update(local_directory=str(batch),created_utc=utc_now(),ssh_connections=1)
    write_json(batch/'submission.json',receipt)
    print('PROBE_JOB='+receipt['job'])
    print('PROBE_PARENT='+parent)
    print('LOCAL_PROBE_DIRECTORY='+str(batch))


def collect(batch):
    batch=approved_directory(batch)
    receipt=json.loads((batch/'submission.json').read_text(encoding='utf-8-sig'))
    job,parent=receipt['job'],receipt['remote_parent']
    if not re.fullmatch(r'\d+[.]pbs\d+',job) or not re.fullmatch(SERVER+r'/records/level4_shared_reset_[0-9a-f]{32}',parent):
        raise ValueError('Invalid submission source')
    destination=batch/('collection_'+uuid.uuid4().hex)
    destination.mkdir(exist_ok=False)
    command=['ssh','-T','NSCC-ASPIRE2A',remote_command(REMOTE_COLLECT,parent,job)]
    write_json(destination/'command.json',{'command':command,'created_utc':utc_now(),'connections':1})
    print('输入一次 NSCC 密码，状态检查与全部文件下载将在同一个 SSH 连接完成。',flush=True)
    process=subprocess.Popen(command,stdout=subprocess.PIPE)
    try:
        verified=receive_files(process.stdout,destination,job,parent)
        if process.wait()!=0:
            raise RuntimeError('Remote verification/source stability failed')
        write_json(destination/'transfer_receipt.json',dict(verified,created_utc=utc_now(),connections=1))
        archive=destination/'diagnostic_evidence.zip'
        if not archive.is_file():
            raise RuntimeError('No diagnostic bundle; logs copied. Inspect collector.log before another run')
        sys.path.insert(0,str(archive_module_root()))
        from level4.archive import unpack_archive
        internal=unpack_archive(archive,destination/'evidence',digest(archive),2*1024**3)
        write_json(destination/'internal_verification.json',internal)
        print('PROBE_EVIDENCE_DIRECTORY='+str(destination/'evidence'))
        exit_path=destination/'job_exit_code.txt'
        print('JOB_EXIT_CODE='+(exit_path.read_text().strip() if exit_path.is_file() else 'unrecorded'))
        print('DIAGNOSTIC_FILES_VERIFIED; CANDIDATE_POOL_NOT_RELEASED')
    except Exception as exc:
        if process.poll() is None:
            process.terminate()
            process.wait()
        write_json(destination/'transfer_failure.json',{'created_utc':utc_now(),'error':str(exc),
                                                       'partial_files_retained':True,'scientific_status_changed':False})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    start=commands.add_parser('submit'); start.add_argument('--delivery',type=Path,required=True)
    finish=commands.add_parser('collect'); finish.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='submit':
        submit(args.delivery.resolve())
    else:
        collect(args.batch)


if __name__=='__main__':
    main()
