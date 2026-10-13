"""Verify downloaded core bodies and restore a real diagnosis using local copies."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from level4.artifacts import read_json, write_json, sha256, utc_now
from level4.archive import verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch
    full = batch / 'core_materials'
    manifest = read_json(full / 'manifest.json')
    verification = verify_files(full / 'files', manifest)
    checkpoint = full / 'files/checkpoint/best.pth'
    expected = 'a05452f681f89ee26763959ec77db234dc0bdcf2cbd2996f7361beebb563acae'
    if sha256(checkpoint) != expected: raise ValueError('Downloaded weight hash differs from the frozen Level3 model')
    # The restore code/input live on D and have their own validated manifest.
    local = Path('D:/SpecCraft/level 4/local_evidence_20261012_01')
    local_manifest = read_json(local / 'manifest.json')
    restore = batch / 'restore_check'
    restore.mkdir(exist_ok=False)
    for sid, folder in [('cpu_tools', 'scripts'), ('CPU_level4', 'level4')]:
        for name, record in local_manifest['files'].items():
            if not name.startswith(sid + '/'): continue
            rel = name[len(sid) + 1:]
            source = local / 'files' / name
            if sha256(source) != record['sha256']: raise ValueError('Frozen local restore code changed: ' + name)
            target = restore / 'runtime' / folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as inp, target.open('xb') as out: shutil.copyfileobj(inp, out)
    shutil.copytree(full / 'files/reset_diagnosis', restore / 'input')
    command = [sys.executable, str(restore / 'runtime/scripts/restore_level4_evidence.py'), '--kind', 'reset',
               '--input', str(restore / 'input'), '--output', str(restore / 'result')]
    executed = subprocess.run(command, cwd=restore, capture_output=True, text=True)
    (restore / 'stdout.txt').write_text(executed.stdout, encoding='utf-8')
    (restore / 'stderr.txt').write_text(executed.stderr, encoding='utf-8')
    write_json(restore / 'command.json', {'command': command, 'exit_code': executed.returncode,
                                       'new_simulation_run': False, 'server_paths_used': False})
    if executed.returncode: raise RuntimeError('D restore check failed; retain logs')
    restoration = read_json(restore / 'result/restore_report.json')
    result = {'kind': 'local_core_backup_verification', 'checked_utc': utc_now(),
              'backup_path': str(batch), 'source_totals': manifest['totals'], 'verification': verification,
              'checkpoint_sha256': expected, 'scientific_status': manifest['run']['scientific_status'],
              'restoration': restoration, 'archive_complete': 'pending_environment_dependency_recovery_review',
              'full_model_execution_restored': False, 'independent_backup': 'not_configured; C and D share Disk 0',
              'gaps': manifest['gaps'], 'paper_materials': [{'kind': 'raw_state_diagnostic',
                                                          'path': str(restore / 'result/restored_state.png')}]}
    write_json(batch / 'verification_report.json', result)
    print(__import__('json').dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
