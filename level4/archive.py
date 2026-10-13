"""Backup operations using the existing artifact hashes/JSON interface.

Never execute a model/simulator, mutate source data, or infer scientific success.
Inventories are retrospective; stability checks do not prove a run has finished.
"""
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import zipfile
from .artifacts import read_json, write_json, sha256, utc_now

EXCLUDED = {'.git', '.ssh', '__pycache__', '.pytest_cache', '.venv', 'venv',
            'node_modules', '.aws', '.azure', '.env', 'credentials',
            'id_rsa', 'id_ed25519', 'id_ecdsa', 'known_hosts', 'authorized_keys'}


def safe_name(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or '\\' in name
            or ':' in name or p.as_posix() != name):
        raise ValueError('Unsafe relative file name: ' + name)
    return p


def excluded(path):
    return any(part.lower() in EXCLUDED or part.lower().endswith(('.pem', '.key'))
               or part.lower().startswith(('.env.', 'access_token', 'password', 'secret_'))
               for part in path.parts)


def signature(path):
    s = Path(path).stat()
    return (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)


def stable_hash(path):
    before = signature(path)
    value = sha256(path)
    if before != signature(path):
        raise ValueError('File changed while hashing: ' + str(path))
    return value, before[0]


def scan_sources(request, hash_files=True):
    """Scan explicit allowlisted roots; links are gaps, never followed silently."""
    entries, gaps, omissions, sources = {}, [], [], []
    ids = set()
    for source in request['sources']:
        sid = source['id']
        if not re.fullmatch(r'[a-zA-Z0-9_.-]+', sid) or sid in ids:
            raise ValueError('Invalid/duplicate source id')
        ids.add(sid)
        root = Path(source['path']).absolute()
        sources.append(dict(source, path=str(root)))
        if not root.exists() or root.is_symlink():
            gaps.append({'source_id': sid, 'path': str(root),
                         'reason': 'symlink_requires_explicit_resolved_body' if root.is_symlink() else 'missing_source',
                         'required': source.get('required', True)})
            continue
        if any(parent.is_symlink() for parent in root.parents):
            gaps.append({'source_id': sid, 'path': str(root), 'reason': 'symlink_parent_requires_resolved_body', 'required': True})
            continue
        def walk(path):
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    rel = child.relative_to(root)
                    if excluded(rel):
                        omissions.append({'source_id': sid, 'path': rel.as_posix(), 'reason': 'credential_or_cache_exclusion'})
                    elif child.is_symlink():
                        gaps.append({'source_id': sid, 'path': rel.as_posix(), 'reason': 'unresolved_symlink', 'required': True})
                    else:
                        yield from walk(child)
            elif path.is_file():
                yield path
            else:
                gaps.append({'source_id': sid, 'path': str(path), 'reason': 'not_regular_file', 'required': True})
        if excluded(Path(root.name)):
            raise ValueError('Excluded source: ' + str(root))
        count = 0
        for path in walk(root):
            rel = path.relative_to(root).as_posix() if root.is_dir() else root.name
            name = sid + '/' + rel
            safe_name(name)
            if hash_files:
                digest, size = stable_hash(path)
            else:
                digest, size = None, path.stat().st_size
            entry = {'size': size, 'source_path': str(path), 'source_mtime_ns': path.stat().st_mtime_ns,
                     'category': source.get('category', 'unspecified')}
            if hash_files:
                entry['sha256'] = digest
            entries[name] = entry
            count += 1
        if not count:
            gaps.append({'source_id': sid, 'path': str(root), 'reason': 'empty_source', 'required': source.get('required', True)})
        expected = source.get('expected_sha256')
        if expected and hash_files:
            if root.is_dir() or entries.get(sid + '/' + root.name, {}).get('sha256') != expected:
                gaps.append({'source_id': sid, 'path': str(root), 'reason': 'expected_hash_mismatch', 'required': True})
    return entries, sources, gaps, omissions


def inventory(request, hash_files=True):
    entries, sources, gaps, omissions = scan_sources(request, hash_files)
    run = dict(request['run'])
    if not run.get('run_id') or run.get('data_kind') not in (
            'synthetic', 'independent_simulation', 'server_preflight', 'code_and_evidence', 'mixed'):
        raise ValueError('Explicit run_id and data_kind required')
    state = run.get('scientific_status', 'unverified')
    if state not in ('completed', 'failed', 'interrupted', 'running', 'unverified', 'not_executed'):
        raise ValueError('Invalid scientific status')
    # Required dependency presence must be explicit; absence cannot become a success.
    for dep in run.get('dependencies', []):
        if dep.get('body_source_id') not in {s['id'] for s in sources}:
            gaps.append({'dependency': dep.get('role'), 'reason': 'reference_only_no_body_source', 'required': dep.get('required', True),
                         'remedy': dep.get('remedy', 'locate_and_copy_existing_body')})
    stable = bool(run.get('stability_basis')) and state in ('completed', 'failed', 'interrupted', 'not_executed')
    required_gaps = [g for g in gaps if g.get('required', True)]
    return {'schema_version': '1.0.0', 'kind': 'experiment_archive_inventory',
            'generated_utc': utc_now(), 'historical_evidence': False,
            'generation_note': 'Retrospective inventory; never substitutes for a historical checksum or exit code.',
            'host': socket.gethostname(), 'run': run, 'sources': sources, 'files': entries,
            'files_sha256': {k: v['sha256'] for k, v in entries.items()} if hash_files else {},
            'totals': {'files': len(entries), 'bytes': sum(v['size'] for v in entries.values())},
            'gaps': gaps, 'exclusions': omissions,
            'statuses': {'source_confirmed': 'passed' if hash_files and not required_gaps else 'incomplete',
                         'archive_complete': 'passed' if hash_files and stable and not required_gaps and run.get('requirements_checked') is True else 'pending',
                         'copied': 'pending', 'verified': 'pending', 'restored': 'pending',
                         'independent_backup': 'not_configured'},
            'stability': {'eligible_for_copy': stable, 'basis': run.get('stability_basis'),
                          'note': 'Rehash and file-set checks at copy time; no claim of scheduler completion.'}}


def check_source(manifest):
    if not manifest['stability']['eligible_for_copy'] or not manifest.get('files_sha256'):
        raise ValueError('Do not archive running/unverified or unhashed sources')
    request = {'run': manifest['run'], 'sources': manifest['sources']}
    entries, _, gaps, _ = scan_sources(request, True)
    expected = manifest['files']
    if set(entries) != set(expected) or gaps != [g for g in manifest['gaps'] if 'source_id' in g]:
        raise ValueError('Source file set/gaps changed; create a new inventory')
    for name, entry in entries.items():
        if (entry['sha256'], entry['size']) != (expected[name]['sha256'], expected[name]['size']):
            raise ValueError('Source content changed: ' + name)


def verify_files(root, manifest, exact=True):
    root = Path(root)
    expected = manifest['files']
    for name, record in expected.items():
        safe_name(name)
        path = root / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
            raise ValueError('Missing file or symlink: ' + name)
        digest, size = stable_hash(path)
        if (digest, size) != (record['sha256'], record['size']):
            raise ValueError('Backup hash/size mismatch: ' + name)
    if exact:
        actual = set()
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError('Symlink in backup')
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        if actual != set(expected):
            raise ValueError('Backup file set differs from manifest')
    return {'status': 'passed', 'checked_utc': utc_now(), 'files': len(expected),
            'bytes': sum(v['size'] for v in expected.values())}


def copy_archive(manifest_path, destination):
    manifest_path, destination = Path(manifest_path), Path(destination).absolute()
    manifest = read_json(manifest_path)
    if destination.exists():
        raise FileExistsError('Backup destination exists; use a new directory')
    for source in manifest['sources']:
        root = Path(source['path']).absolute()
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError('Source and destination must be separate')
    check_source(manifest)
    if shutil.disk_usage(destination.parent).free < manifest['totals']['bytes'] + 16 * 1024**2:
        raise ValueError('Insufficient destination free space')
    destination.mkdir()
    try:
        for name, record in manifest['files'].items():
            safe_name(name)
            target = destination / 'files' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with Path(record['source_path']).open('rb') as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
        check_source(manifest)
        validation = verify_files(destination / 'files', manifest)
        with manifest_path.open('rb') as inp, (destination / 'manifest.json').open('xb') as out:
            shutil.copyfileobj(inp, out)
        receipt = {'kind': 'backup_transfer_receipt', 'copied_utc': utc_now(),
                   'manifest_sha256': sha256(manifest_path), 'destination': str(destination),
                   'scientific_status': manifest['run'].get('scientific_status'),
                   'statuses': dict(manifest['statuses'], copied='passed', verified='passed'),
                   'verification': validation, 'independent_copy': False}
        write_json(destination / 'transfer_receipt.json', receipt)
        return receipt
    except Exception as exc:
        write_json(destination / 'transfer_failure.json', {'status': 'failed', 'utc': utc_now(), 'error': str(exc),
                   'scientific_status': manifest['run'].get('scientific_status'), 'partial_files_retained': True})
        raise


def pack_archive(backup, output):
    backup, output = Path(backup), Path(output)
    manifest = read_json(backup / 'manifest.json')
    verify_files(backup / 'files', manifest)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name in ('manifest.json', 'transfer_receipt.json'):
            z.write(backup / name, name)
        for name in manifest['files']:
            z.write(backup / 'files' / name, 'files/' + name)
    # Check internal members and hashes, not just ZIP presence/CRC.
    validation = verify_zip(output, manifest['totals']['bytes'] + 16 * 1024**2)
    result = {'created_utc': utc_now(), 'archive': str(output), 'bytes': output.stat().st_size,
              'sha256': sha256(output), 'internal_verification': validation}
    write_json(output.with_suffix(output.suffix + '.receipt.json'), result)
    return result


def verify_zip(path, max_bytes=16 * 1024**3):
    import hashlib
    import json
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        if sum(i.file_size for i in z.infolist()) > max_bytes:
            raise ValueError('ZIP exceeds explicit uncompressed budget')
        for info in z.infolist():
            safe_name(info.filename)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('ZIP directory/symlink members unsupported')
        # An inventory report may contain manifest.json describing remote
        # sources, plus evidence_manifest.json hashing the actual ZIP bodies.
        # The transport manifest is authoritative for this package's members.
        if 'evidence_manifest.json' in names:
            manifest = json.loads(z.read('evidence_manifest.json'))
            expected = {name: {'sha256': v} for name, v in manifest['files_sha256'].items()}
            allowed = set(expected) | {'evidence_manifest.json'}
            manifest_role = 'evidence_package_files'
        elif 'manifest.json' in names and 'files' in json.loads(z.read('manifest.json')):
            manifest = json.loads(z.read('manifest.json'))
            expected = {'files/' + name: v for name, v in manifest['files'].items()}
            allowed = set(expected) | {'manifest.json', 'transfer_receipt.json'}
            manifest_role = 'backup_payload_files'
        else:
            raise ValueError('No supported internal manifest')
        if set(names) != allowed:
            raise ValueError('ZIP members differ from internal manifest')
        for name, record in expected.items():
            safe_name(name)
            h = hashlib.sha256()
            with z.open(name) as inp:
                for chunk in iter(lambda: inp.read(1024**2), b''):
                    h.update(chunk)
            if h.hexdigest() != record['sha256'] or ('size' in record and z.getinfo(name).file_size != record['size']):
                raise ValueError('ZIP internal hash/size mismatch: ' + name)
    return {'status': 'passed', 'checked_utc': utc_now(), 'files': len(expected), 'manifest_role': manifest_role}


def unpack_archive(archive, output, expected_sha256, max_bytes=16 * 1024**3):
    archive, output = Path(archive), Path(output)
    if sha256(archive) != expected_sha256:
        raise ValueError('Transport archive hash mismatch')
    verified = verify_zip(archive, max_bytes)
    if output.exists():
        raise FileExistsError('Restore destination exists')
    if shutil.disk_usage(output.parent).free < max_bytes:
        raise ValueError('Insufficient space for explicit extraction budget')
    output.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
    return {'status': 'unpacked_and_verified_not_restored', 'internal_verification': verified,
            'output': str(output), 'archive_sha256': expected_sha256}


def check_pool_records(pool):
    """Separate evidence coverage from scientific decisions; no truth sent to selector."""
    pool = Path(pool)
    manifest = read_json(pool / 'pool.json')
    required = ['planner_input.npz', 'grid.npz', 'model_config.yaml', 'source_scene.yaml',
                'source_metadata.json', 'pool_config.json', 'runtime_snapshot/manifest.json']
    missing = [name for name in required if not (pool / name).is_file()]
    candidates = []
    for row in manifest['candidates']:
        cid = row['candidate_id']
        safe_name(cid)
        actual, predicted = pool / 'executions' / cid, pool / 'predictions' / cid
        execution = 'not_executed'
        if actual.exists():
            execution = ('completed' if (actual / 'complete.json').is_file() else
                         'failed' if (actual / 'failure.json').is_file() else
                         'process_exited' if (actual / 'exit.json').is_file() else 'unverified_or_running')
        candidates.append({'candidate_id': cid, 'execution_status': execution,
                           'prediction_status': 'recorded' if (predicted / 'complete.json').is_file() else 'not_recorded',
                           'actual_fields_from_prediction': False})
    return {'kind': 'pool_evidence_coverage', 'generated_utc': utc_now(), 'missing': missing,
            'checkpoint_body_available_at_original_path': Path(manifest.get('checkpoint', '__missing__')).is_file(),
            'external_dependency_archive': 'required_even_if_original_path_exists', 'candidates': candidates,
            'recording_contract': {'raw_material_logs': 'scenes/0000/log.pkl for every executed candidate including failures',
                                   'prediction_geometry': 'trajectory.npz', 'query_grid': 'grid.npz',
                                   'no_extra_simulations': True}}


def record_run(run_dir, pool, exit_code, output):
    """Prospective post-run small-evidence snapshot; failure cannot become success."""
    run_dir, pool, output = Path(run_dir), Path(pool), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    coverage = check_pool_records(pool) if (pool / 'pool.json').is_file() else {'pool_status': 'not_prepared'}
    coverage['reset_pools'] = [check_pool_records(p.parent) for p in sorted(run_dir.glob('reset_*/pool/pool.json'))]
    write_json(output / 'coverage.json', coverage)
    paths = [p for p in run_dir.rglob('*') if p.is_file() and not p.is_symlink()
             and not excluded(p.relative_to(run_dir)) and p.name not in ('backup_result.txt', 'backup_exit_code.txt')
             and p.suffix in ('.json', '.yaml', '.txt', '.log', '.patch')]
    # A caller's completion is authoritative only for this new attempt. No historical inference.
    request = {'run': {'run_id': run_dir.name, 'level': 4, 'task': 'see_pool_config', 'method': 'fixed_pool',
                       'data_kind': 'mixed', 'scientific_status': 'completed' if exit_code == 0 else 'failed',
                       'exit_code': exit_code, 'stability_basis': 'EXIT trap called after stage subprocesses returned',
                       'requirements_checked': False, 'scope': 'small_evidence_only',
                       'dependencies': [{'role': 'raw_trajectories_weights_inputs', 'required': True,
                                         'remedy': 'stage_end_full_inventory_and_copy'}]},
               'sources': [{'id': 'evidence_%06d' % i, 'path': str(p), 'category': 'small_evidence'}
                           for i, p in enumerate(sorted(paths))]}
    manifest = inventory(request)
    write_json(output / 'manifest.json', manifest)
    # No source overwrite; copies go to a sibling evidence directory.
    return copy_archive(output / 'manifest.json', output / 'small_backup')
