"""Read-only inventory and exclusive copy/verification; never starts experiments."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from level4.artifacts import read_json, write_json
from level4.archive import (inventory, copy_archive, verify_files, pack_archive,
                            verify_zip, unpack_archive, check_pool_records, record_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scan = sub.add_parser('inventory')
    scan.add_argument('--request', type=Path, required=True)
    scan.add_argument('--output', type=Path, required=True)
    scan.add_argument('--metadata-only', action='store_true', help='Capacity first; cannot copy this unhashed inventory')
    copy = sub.add_parser('copy')
    copy.add_argument('--manifest', type=Path, required=True)
    copy.add_argument('--output', type=Path, required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--backup', type=Path, required=True)
    pack = sub.add_parser('pack')
    pack.add_argument('--backup', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    zipcheck = sub.add_parser('verify-zip')
    zipcheck.add_argument('--archive', type=Path, required=True)
    unpack = sub.add_parser('unpack')
    unpack.add_argument('--archive', type=Path, required=True)
    unpack.add_argument('--output', type=Path, required=True)
    unpack.add_argument('--sha256', required=True)
    unpack.add_argument('--max-bytes', type=int, required=True)
    poolcheck = sub.add_parser('check-pool')
    poolcheck.add_argument('--pool', type=Path, required=True)
    poolcheck.add_argument('--output', type=Path, required=True)
    poolcheck.add_argument('--strict', action='store_true', help='Reject missing core records before experiment stages')
    finish = sub.add_parser('record-run')
    finish.add_argument('--run-dir', type=Path, required=True)
    finish.add_argument('--pool', type=Path, required=True)
    finish.add_argument('--exit-code', type=int, required=True)
    finish.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inventory':
        result = inventory(read_json(args.request), not args.metadata_only)
        write_json(args.output, result)
        result = {k: result[k] for k in ('totals', 'statuses', 'gaps', 'stability')}
    elif args.command == 'copy':
        result = copy_archive(args.manifest, args.output)
    elif args.command == 'verify':
        result = verify_files(args.backup / 'files', read_json(args.backup / 'manifest.json'))
    elif args.command == 'pack':
        result = pack_archive(args.backup, args.output)
    elif args.command == 'verify-zip':
        result = verify_zip(args.archive)
    elif args.command == 'unpack':
        result = unpack_archive(args.archive, args.output, args.sha256, args.max_bytes)
    elif args.command == 'check-pool':
        result = check_pool_records(args.pool)
        write_json(args.output, result)
        if args.strict and result['missing']:
            raise ValueError('Core evidence inputs missing: ' + ', '.join(result['missing']))
    else:
        result = record_run(args.run_dir, args.pool, args.exit_code, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'backup_status': 'failed', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(2)
