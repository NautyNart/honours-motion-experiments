import argparse
from pathlib import Path
import json
import sys
import zipfile
from .common import REPO, read, now, error_record
from .experiment import validate_config

def main():
    parser = argparse.ArgumentParser(description='Ground-contact assessment and correction pilot')
    parser.add_argument('--work', type=Path, default=Path('outputs/physical-pilot'))
    parser.add_argument('--config', type=Path, default=REPO / 'pilot/configs/pilot.json')
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--baseline', type=Path)
    prep.add_argument('--source-index', type=Path)
    sub.add_parser('measure')
    sub.add_parser('corrupt')
    sub.add_parser('reconstruct')
    sub.add_parser('generate')
    export = sub.add_parser('review-export')
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--mode', choices=['clips', 'pairs'], default='clips')
    export.add_argument('--run-id')
    imp = sub.add_parser('review-import')
    imp.add_argument('file', type=Path)
    imp.add_argument('--mapping', type=Path)
    run = sub.add_parser('refine')
    run.add_argument('--run-id', required=True)
    rep = sub.add_parser('report')
    rep.add_argument('--run-id')
    bundle = sub.add_parser('bundle')
    bundle.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    config = read(args.config)
    validate_config(config)
    try:
        if args.command == 'prepare':
            from .data import prepare, make_source_template
            rows = prepare(work, config, args.baseline, args.source_index)
            template = work / 'source_index_template.csv'
            if not template.exists():
                make_source_template(template)
            result = {'slots': len(rows), 'available': sum(bool(r['paths']) for r in rows)}
        elif args.command == 'corrupt':
            from .data import corrupt
            corrupt(work)
            result = {'manifest': str(work / 'manifest.json')}
        elif args.command in ('reconstruct', 'generate'):
            from .experiment import gpu_prepare
            result = gpu_prepare(work, config, args.command)
        elif args.command == 'measure':
            from .experiment import measure_all
            records = measure_all(work, config)
            result = {'records': len(records), 'errors': sum('error' in r for r in records)}
        elif args.command == 'review-export':
            from .review import export_review
            result = {'review_entries': export_review(work, config, args.output, args.mode, args.run_id),
                      'viewer': str(args.output.resolve() / 'index.html')}
        elif args.command == 'review-import':
            from .review import import_review
            result = {'imported_reviews': import_review(work, args.file, config, args.mapping)}
        elif args.command == 'refine':
            from .experiment import refine
            result = refine(work, config, args.run_id)
        elif args.command == 'report':
            from .report import report
            result = report(work, config, args.run_id)
        else:
            target = args.output.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            # Never package arbitrary repo files, credentials, checkpoints or Python environments.
            allowed = {'.json', '.jsonl', '.csv', '.npy', '.npz', '.gif', '.svg', '.html', '.md', '.txt'}
            files = [p for p in work.rglob('*') if p.is_file() and p.suffix in allowed and p != target]
            with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                for path in files:
                    archive.write(path, path.relative_to(work))
            result = {'bundle': str(target), 'files': len(files), 'bytes': target.stat().st_size}
        print(json.dumps(result, indent=2))
        log = work / 'commands.jsonl'
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': now(), 'command': args.command, 'result': result}) + '\n')
        if result.get('errors'):
            return 1
        return 0
    except Exception as exc:
        error_record(work, args.command, 'command', exc)
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
