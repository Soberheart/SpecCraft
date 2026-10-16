"""Regression for the 2026-10-16 ``JSONDecodeError: Invalid \\escape`` failures.

Root cause (L4-01 A2): Level 4 writers emit UTF-8 JSON with ``ensure_ascii=False``;
several tests (and a few readers) decoded those files with the locale default
encoding.  On the Windows host the default is cp936/GBK and the user directory
``Admin（无密码）`` is non-ASCII: the trailing byte 0x89 of ``）`` (EF BC 89) becomes a
GBK lead byte, swallows the first backslash of the JSON escape ``\\\\`` and leaves an
invalid ``\\D``/``\\.`` escape.  The fix is explicit ``encoding='utf-8'`` on text I/O;
writers were already correct, so no writer or historical file is changed.
"""
import ast
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PATH = 'C:\\Users\\Admin（无密码）\\Documents\\ChatGPT\\Doughnet+LLM\\Doughnet-LLM\\x.json'

# Text I/O without explicit encoding that is deliberately left unchanged.  Frozen
# probe/review scripts were delivered to NSCC under recorded SHA-256 and only run in
# the server's UTF-8 locale; editing them would break their delivery identity.
ALLOWED_DEFAULT_ENCODING = {
    ('scripts/probe_level4_cuda.py', 'maps.read_text()'): 'Linux /proc/self/maps, ASCII library paths',
    ('scripts/probe_level4_clock_diagnostic.py', "(tools/'tool_manifest.json').read_text()"): 'frozen NSCC tool, ASCII manifest',
    ('scripts/probe_level4_pending_candidate.py', "(tools/'tool_manifest.json').read_text()"): 'frozen NSCC tool, ASCII manifest',
    ('scripts/probe_level4_transaction.py', 'request_path.read_text()'): 'frozen NSCC tool, server request',
    ('scripts/probe_level4_initialization.py', "(output / ('attempt_%d.log' % attempt)).open('x')"): 'subprocess log handle only',
    ('scripts/review_level4_clock_evidence.py', "(collection / 'job_exit_code.txt').read_text()"): 'historical review, ASCII exit code',
}


def default_encoding_calls(source):
    """Return text-mode read_text/write_text/open calls lacking ``encoding=``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ('read_text', 'write_text'):
            mode = None
        elif isinstance(func, ast.Attribute) and func.attr == 'open':
            first = node.args[0] if node.args else None
            if first is not None and not (isinstance(first, ast.Constant) and isinstance(first.value, str)
                                          and set(first.value) <= set('rwxab+t')):
                continue  # ZipFile.open(member) and similar, not Path.open(mode)
            mode = first
        elif isinstance(func, ast.Name) and func.id == 'open':
            mode = node.args[1] if len(node.args) > 1 else None
        else:
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        if 'encoding' in keywords:
            continue
        mode = keywords.get('mode', mode)
        if isinstance(mode, ast.Constant) and 'b' in mode.value:
            continue
        found.append((node.lineno, ast.get_source_segment(source, node)))
    return found


class TextEncodingTests(unittest.TestCase):
    def test_level4_sources_tests_and_scripts_use_explicit_encoding(self):
        files = sorted(ROOT.glob('level4/*.py')) + sorted(ROOT.glob('scripts/*.py')) + \
            sorted(ROOT.glob('tests/test_level4_*.py'))
        self.assertGreater(len(files), 40)
        offenders, used = [], set()
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            for line, segment in default_encoding_calls(path.read_text(encoding='utf-8')):
                if (name, segment) in ALLOWED_DEFAULT_ENCODING:
                    used.add((name, segment))
                else:
                    offenders.append('%s:%d %s' % (name, line, segment))
        self.assertEqual(offenders, [])
        self.assertEqual(used, set(ALLOWED_DEFAULT_ENCODING))  # stale allowance must be removed

    def test_lint_detects_default_reads_and_ignores_binary_or_explicit(self):
        sample = ("p.read_text()\np.read_text(encoding='utf-8')\np.write_text(s)\n"
                  "p.open('rb')\np.open('x')\nz.open(name)\nopen(f)\nopen(f, 'wb')\n")
        self.assertEqual([line for line, _ in default_encoding_calls(sample)], [1, 3, 5, 7])

    def test_windows_user_path_json_is_utf8_and_gbk_decoding_reproduces_failure(self):
        from scripts import level4_remote_session as remote
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'internal_verification.json'
            remote.write_json(path, {'evidence': WINDOWS_PATH})
            data = path.read_bytes()
        self.assertIn('）'.encode('utf-8') + b'\\\\', data)
        self.assertEqual(json.loads(data.decode('utf-8'))['evidence'], WINDOWS_PATH)
        with self.assertRaisesRegex(json.JSONDecodeError, 'Invalid \\\\escape'):
            json.loads(data.decode('gbk'))


if __name__ == '__main__':
    unittest.main()
