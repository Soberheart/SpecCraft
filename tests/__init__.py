"""Level 4 test package.

A regular package, so ``from tests.test_level4_... import`` resolves to this
repository even when the environment ships an unrelated top-level ``tests``
package (found on NSCC 2026-10-17: 5 import errors without this file).
"""
