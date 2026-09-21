"""Every shipped config composes, and a mistake in one is caught before a run starts.

The configs are the interface most people touch, and a broken one fails after the data has loaded rather than before.
Composing all of them here costs a fraction of a second and catches the cases that matter: a mode file that names a
key the schema dropped, a dataset file missing a required path, and the typo that used to be caught by a hand-written
did-you-mean check.
"""

import itertools
import os
import unittest
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from config import EVALUATION_MODES, RunConfig, validate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")

DATASETS = ("novamag", "mp")
MODES = ("cross_validation", "random_split", "ood", "uncertainty", "predict")


def composed(*overrides: str) -> RunConfig:
    """Compose the shipped config with the given `key=value` overrides, exactly as main.py would."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return cast(RunConfig, OmegaConf.to_object(compose(config_name="config", overrides=list(overrides))))


class ShippedConfigTests(unittest.TestCase):
    """The ten dataset x mode combinations all produce a valid RunConfig."""

    def test_every_dataset_and_mode_composes(self):
        for dataset, mode in itertools.product(DATASETS, MODES):
            with self.subTest(dataset=dataset, mode=mode):
                cfg = composed(f"dataset={dataset}", f"mode={mode}")
                validate(cfg)
                self.assertEqual(cfg.evaluation_mode, mode)
                if mode != "predict":
                    self.assertTrue(cfg.models, "an evaluation mode must name at least one model")

    def test_every_evaluation_mode_has_a_mode_file(self):
        """A mode main.py can dispatch to but no file selects is unreachable from the command line."""
        self.assertEqual(set(MODES), set(EVALUATION_MODES))

    def test_the_dataset_file_carries_the_per_dataset_keys(self):
        novamag, mp = composed("dataset=novamag", "mode=ood"), composed("dataset=mp", "mode=ood")
        self.assertNotEqual(novamag.dataset_path, mp.dataset_path)
        self.assertNotEqual(novamag.output_dir, mp.output_dir)


class BrokenConfigTests(unittest.TestCase):
    """The mistakes people actually make are refused with the offending key named."""

    def test_an_unknown_key_is_rejected(self):
        with self.assertRaises(ConfigCompositionException) as caught:
            composed("dataset=mp", "mode=ood", "ood.kk=5")
        self.assertIn("ood.kk", str(caught.exception))

    def test_a_wrong_type_is_rejected(self):
        with self.assertRaises((ConfigCompositionException, ValidationError)) as caught:
            composed("dataset=mp", "mode=ood", "kfold.folds=ten")
        self.assertIn("kfold.folds", str(caught.exception))

    def test_an_unknown_ood_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            validate(composed("dataset=mp", "mode=ood", "ood.scenarios=[element,elemnt]"))

    def test_a_subset_of_scenarios_can_be_chosen(self):
        cfg = composed("dataset=mp", "mode=ood", "ood.scenarios=[element,period]")
        self.assertEqual(list(cfg.ood.scenarios), ["element", "period"])

    def test_an_override_reaches_the_nested_field(self):
        """The whole point of the layout: a run varies one key without a new file."""
        cfg = composed("dataset=mp", "mode=ood", "ood.k=3", "kfold.seeds=[7]")
        self.assertEqual(cfg.ood.k, 3)
        self.assertEqual(list(cfg.kfold.seeds), [7])


if __name__ == "__main__":
    unittest.main()
