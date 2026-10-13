"""CPU archive of existing Level4 dependencies; no training/model/simulation."""
import argparse
from pathlib import Path
import subprocess
import sys
import re
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from level4.artifacts import read_json, write_json, sha256, utc_now
from level4.archive import inventory, copy_archive, pack_archive


def scheduler_record(job):
    result = subprocess.run(['qstat', '-xf', job], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Cannot confirm job ended: ' + job + ': ' + result.stdout + result.stderr)
    return result.stdout + result.stderr


def backup_sources(request, output, scheduler=scheduler_record):
    output = Path(output).absolute()
    if not request.get('copy_budget', {}).get('destination_confirmation_required') is False:
        raise ValueError('Destination confirmation not recorded')
    output.mkdir(parents=True, exist_ok=False)
    try:
        job_records = {}
        for job, prior in request['run']['source_science_states'].items():
            text = scheduler(job)
            (output / ('qstat_' + job.split('.')[0] + '.txt')).write_text(text, encoding='utf-8')
            state = re.search(r'^\s*job_state = (\S+)\s*$', text, re.M)
            exit_code = re.search(r'^\s*Exit_status = (-?\d+)\s*$', text, re.M)
            if not state or state.group(1) != 'F' or not exit_code:
                raise ValueError('Job is not confirmed ended: ' + job)
            if int(exit_code.group(1)) != prior['exit_code']:
                raise ValueError('Job exit record differs from reviewed inventory: ' + job)
            job_records[job] = {'job_state': 'F', 'exit_code': int(exit_code.group(1)),
                                'qstat_sha256': sha256(output / ('qstat_' + job.split('.')[0] + '.txt'))}
        write_json(output / 'qstat_review.json', {'checked_utc': utc_now(), 'jobs': job_records})
        request['run']['stability_basis'] = 'Rechecked retained F scheduler states at ' + utc_now() + '; source hashed before and after copy'
        request['run']['source_science_states'] = job_records
        write_json(output / 'request.json', request)
        full = inventory(request)
        budget = request['copy_budget']['maximum_uncompressed_bytes']
        if full['totals']['bytes'] > budget:
            raise ValueError('Source capacity exceeds approved 1 GiB budget; retain inventory and review')
        bad_sources = [g for g in full['gaps'] if 'source_id' in g and g.get('required', True)]
        if bad_sources:
            raise ValueError('Source body/link/hash gaps: ' + str(bad_sources))
        write_json(output / 'source_manifest.json', full)
        # Copy small scientific records FIRST. No binary states or model weights.
        small_sources = []
        for name, row in full['files'].items():
            rel = name.split('/', 1)[-1]
            if (row['size'] <= 1024**2 and Path(rel).suffix.lower() in ('.json', '.yaml', '.yml', '.log', '.txt', '.patch', '.md')
                    and '/files/' not in '/' + rel):
                small_sources.append({'id': 'small_%06d' % len(small_sources), 'path': row['source_path'],
                                      'category': 'small_evidence', 'original_member': name})
        small_sources.extend({'id': 'scheduler_' + job.split('.')[0],
                              'path': str(output / ('qstat_' + job.split('.')[0] + '.txt')), 'category': 'scheduler'}
                             for job in job_records)
        small_request = {'run': dict(request['run'], run_id=request['run']['run_id'] + '_small',
                         scope='small_evidence_only', dependencies=[{'role': 'full_raw_weights_inputs', 'required': True,
                                                                    'remedy': 'refer_to_core_bundle'}]),
                         'sources': small_sources}
        small = inventory(small_request)
        write_json(output / 'small_manifest.json', small)
        copy_archive(output / 'small_manifest.json', output / 'small_backup')
        small_pack = pack_archive(output / 'small_backup', output / 'small_evidence.zip')
        copy_archive(output / 'source_manifest.json', output / 'core_backup')
        core_pack = pack_archive(output / 'core_backup', output / 'core_materials.zip')
        receipt = {'kind': 'level4_core_materials_backup_receipt', 'created_utc': utc_now(), 'server_parent': str(output),
                   'scientific_status': request['run']['scientific_status'], 'source_manifest_sha256': sha256(output / 'source_manifest.json'),
                   'source_totals': full['totals'], 'gaps': full['gaps'], 'source_state_modified': False,
                   'simulation_run': False, 'training_run': False, 'all_known_jobs_ended': True,
                   'independent_backup': 'not_configured', 'full_environment_restorable': False,
                   'bundles': [{'id': 'small_evidence', 'archive': small_pack['archive'], 'sha256': small_pack['sha256'],
                                'bytes': small_pack['bytes'], 'max_uncompressed_bytes': small['totals']['bytes'] + 16*1024**2},
                               {'id': 'core_materials', 'archive': core_pack['archive'], 'sha256': core_pack['sha256'],
                                'bytes': core_pack['bytes'], 'max_uncompressed_bytes': budget + 16*1024**2}]}
        write_json(output / 'receipt.json', receipt)
        return receipt
    except Exception as exc:
        write_json(output / 'archive_failure.json', {'backup_status': 'failed', 'recorded_utc': utc_now(), 'error': str(exc),
                    'scientific_status': request['run'].get('scientific_status'), 'partial_files_retained': True,
                    'simulation_run': False, 'training_run': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = backup_sources(read_json(args.request), args.output)
    print('CORE_RECEIPT_JSON=' + __import__('json').dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
