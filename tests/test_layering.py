"""Pins the one architectural rule the code cannot state for itself: where the run config stops.

The library is meant to be importable from a notebook or another project that has never heard of this repo's YAML
schema. Nothing enforces that at run time — a single `from config import ...` in `evaluate/` would work fine and
quietly tie the scoring code to one CLI's config file. This test is the enforcement.

`RUNNERS_STILL_READING_CONFIG` is the list of modules not yet converted. Converting one means moving its config
handling into `examples/` and deleting its line here, so the remaining work is visible rather than remembered.

The same rule is what PyKale, WILDS and detectron2 follow: config-driven code lives in the runner layer, and the
library below it takes explicit arguments.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that hold neither library nor runner code.
IGNORED_DIRECTORIES = ("temp", "results", "figures", "data", "plots", "tests", ".git")

# The runner layer, which exists to read a config.
CONFIG_READERS = ("main.py", "config.py")

# Runners that still live under pipeline/ and still read a config. Shrinks to nothing as each mode is converted.
RUNNERS_STILL_READING_CONFIG = (
    "pipeline/random_split_pipeline.py",
    "pipeline/ood_pipeline.py",
    "pipeline/ood_scenarios.py",
    "pipeline/uncertainty_pipeline.py",
    "pipeline/inference_pipeline.py",
    "pipeline/data_visualization.py",
)


def imported_module_names(path: Path):
    """Yield every module name the file imports, for both `import x` and `from x import y`."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def library_modules():
    """Every .py file that is meant to work without a run config."""
    allowed = set(CONFIG_READERS) | set(RUNNERS_STILL_READING_CONFIG)
    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        if relative.parts[0] in IGNORED_DIRECTORIES or relative.parts[0] == "examples":
            continue
        if relative.as_posix() in allowed:
            continue
        yield relative, path


class ConfigBoundaryTests(unittest.TestCase):
    """`config` is a runner concern, not a library concern."""

    def test_library_modules_do_not_import_config(self):
        offenders = [
            f"{relative} imports {name}"
            for relative, path in library_modules()
            for name in imported_module_names(path)
            if name == "config" or name.startswith("config.")
        ]
        self.assertEqual(offenders, [], "library modules must take explicit arguments, not a RunConfig")

    def test_the_listed_runners_still_exist(self):
        """Guards the exemptions: a renamed or deleted runner must be removed from the list, not left stale."""
        for relative in RUNNERS_STILL_READING_CONFIG:
            self.assertTrue((REPO_ROOT / relative).is_file(), f"{relative} is listed as exempt but does not exist")

    def test_something_is_actually_being_checked(self):
        """Guards the test itself: a bad filter would make the check above pass by scanning nothing."""
        checked = {relative.as_posix() for relative, _ in library_modules()}
        self.assertIn("evaluate/cross_validation.py", checked)
        self.assertIn("pipeline/cross_validation.py", checked)
        self.assertGreater(len(checked), 20)


if __name__ == "__main__":
    unittest.main()
