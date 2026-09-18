"""Pins the one architectural rule the code cannot state for itself: where the run config stops.

The stage packages are meant to be importable from a notebook or another project that has never heard of this repo's
YAML schema. Nothing enforces that at run time — a single `from config import ...` in `evaluate/` would work fine and
quietly tie the scoring code to one CLI's config file. This test is the enforcement.

The same rule is what PyKale, WILDS and detectron2 follow: config-driven code lives in the runner layer, and the
library below it takes explicit arguments.
"""

import ast
import unittest
from pathlib import Path

# Packages that must stay usable without a RunConfig. `pipeline/` is deliberately absent: it is the layer that
# translates a config into arguments, the way PyKale's examples/*/main.py and detectron2's from_config do.
STAGE_PACKAGES = ("loaddata", "prepdata", "predict", "evaluate", "interpret", "utils")

REPO_ROOT = Path(__file__).resolve().parent.parent


def imported_module_names(path: Path):
    """Yield every module name the file imports, for both `import x` and `from x import y`."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


class ConfigBoundaryTests(unittest.TestCase):
    """`config` is a runner concern, not a stage concern."""

    def test_stage_packages_do_not_import_config(self):
        offenders = [
            f"{path.relative_to(REPO_ROOT)} imports {name}"
            for package in STAGE_PACKAGES
            for path in (REPO_ROOT / package).rglob("*.py")
            for name in imported_module_names(path)
            if name == "config" or name.startswith("config.")
        ]
        self.assertEqual(offenders, [], "stage packages must take explicit arguments, not a RunConfig")

    def test_the_packages_being_checked_actually_exist(self):
        """Guards the test itself: a renamed package would make the check above pass by scanning nothing."""
        for package in STAGE_PACKAGES:
            self.assertTrue((REPO_ROOT / package / "__init__.py").is_file(), f"{package}/ is missing")


if __name__ == "__main__":
    unittest.main()
