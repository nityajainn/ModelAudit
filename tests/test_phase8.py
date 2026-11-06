"""Failure cases for the release checks themselves."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/check_release.py"
spec = importlib.util.spec_from_file_location("release_checks", path)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class ReleaseChecks(unittest.TestCase):
    def report(self, lines=95, branches=90, child_misses=0):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "coverage.json"
            report.write_text(
                json.dumps(
                    {
                        "meta": {"branch_coverage": True},
                        "totals": {
                            "num_statements": 100,
                            "covered_lines": lines,
                            "num_branches": 100,
                            "covered_branches": branches,
                        },
                        "files": {
                            "src/modelaudit/__main__.py": {
                                "summary": {"covered_lines": 1, "missing_lines": child_misses}
                            }
                        },
                    }
                )
            )
            checks.check_coverage(report)

    def test_line_and_combined_gates_are_independent(self):
        self.report(90, 90)
        for lines, branches in [(89, 100), (99, 80)]:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                self.report(lines, branches)

    def test_child_entry_must_be_exercised(self):
        with self.assertRaises(ValueError):
            self.report(child_misses=1)

    def test_unsafe_archive_names_fail(self):
        for name in (
            "../outside",
            "pkg/../escape",
            "/tmp/file",
            "c:/absolute",
            "p\\a",
            "p//a",
            "x:stream",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                checks.safe_name(name)

    def test_archive_cannot_drop_code_or_include_local_data(self):
        for actual in (set(), {"modelaudit/cli.py", "data/customers.csv"}):
            with self.assertRaises(ValueError):
                checks.verify_members(actual, {"modelaudit/cli.py"})
