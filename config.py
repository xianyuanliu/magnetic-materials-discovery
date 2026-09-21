"""The configuration schema for the magnetic materials discovery experiments."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore

EVALUATION_MODES = ("predict", "random_split", "cross_validation", "ood", "uncertainty")

OOD_SCENARIOS = ("element", "period", "group", "system", "cluster", "sparsex", "sparsey")


@dataclass
class KFoldConfig:
    """Partition the rows into K folds."""

    folds: int = 5
    shuffle: bool = True
    seeds: Tuple[int, ...] = (0,)


@dataclass
class RandomSplitConfig:
    """Draw one random train/valid split per seed."""

    seeds: Tuple[int, ...] = (0,)
    train_size: float = 0.8


@dataclass
class TuningConfig:
    """Search hyperparameters inside every fold."""

    enabled: bool = False
    cv_folds: int = 3
    n_iter: int = 20


@dataclass
class InterpretConfig:
    """Choose the models to be interpreted and compared in the case studies."""

    enabled: bool = False
    explain_models: Tuple[str, ...] = ()
    case_study_models: Tuple[str, ...] = ()


@dataclass
class OODConfig:
    """Build the out-of-distribution splits.

    Attributes:
        k: KMeans clusters for the LOCO scenario.
        max_splits: Maximum number of splits.
        elements, periods, groups: Hold-out targets.
        systems: Element sets held out together, written "Fe-Co".
        fractions: Held-out fractions for SparseX and SparseY.
        size_matched_control: Score a random split of the same sizes beside each shifted split.
    """

    scenarios: Tuple[str, ...] = OOD_SCENARIOS
    k: int = 10
    cluster_seed: int = 0
    seeds: Tuple[int, ...] = (0,)
    max_splits: Optional[int] = None
    elements: Optional[Tuple[str, ...]] = None
    periods: Optional[Tuple[int, ...]] = None
    groups: Optional[Tuple[int, ...]] = None
    systems: Tuple[str, ...] = ()
    fractions: Tuple[float, ...] = (0.1, 0.2)
    sparsex_neighbors: int = 5
    sparsey_center: str = "median"
    min_test: int = 20
    min_train: int = 50
    period_strict: bool = False
    group_strict: bool = False
    size_matched_control: bool = True


@dataclass
class UncertaintyConfig:
    """Calibrate the prediction intervals of one model.

    Attributes:
        model: One of pipeline.uncertainty.ENSEMBLE_STD_MODELS.
        alpha: Nominal miscoverage; 0.05 asks for 95% intervals.
        calibration_fraction: Share of each training set withheld to calibrate the conformal intervals.
    """

    model: str = "rf"
    alpha: float = 0.05
    calibration_fraction: float = 0.25
    seeds: Tuple[int, ...] = (0,)


@dataclass
class PredictConfig:
    """Load one model and predict new compositions.

    Attributes:
        model: Model to fit when `model_path` has no bundle; None uses the one an evaluation run saved.
        retrain: Refit even if `model_path` exists.
        input_path, formulas: A CSV with the formula column, or an inline list.
    """

    model: Optional[str] = None
    model_path: Optional[str] = None
    retrain: bool = False
    input_path: Optional[str] = None
    formulas: Tuple[str, ...] = ()


@dataclass
class RunConfig:
    """Setup and run a complete experiment.

    Attributes:
        dataset: Dataset name in the lowercase.
        pt_path, mm_path: Paths to the periodic-table and Miedema reference spreadsheets.
        feature_columns: Column names to use in the experiment.
        compare_models: The models to be compared.
        model_random_state: Random state for model construction and tuning.
    """

    dataset: str
    dataset_path: str
    evaluation_mode: str
    pt_path: str
    mm_path: str

    target_column: str = "saturation magnetization"
    formula_column: str = "chemical formula"
    feature_columns: Optional[Tuple[str, ...]] = None

    models: Tuple[str, ...] = ()
    compare_models: Optional[Tuple[str, str]] = None

    enable_data_visualization: bool = False
    model_random_state: int = 0
    plots_output_dir: str = "./plots"
    output_dir: str = "./results"
    save_results: bool = True
    print_results: bool = True

    kfold: KFoldConfig = field(default_factory=KFoldConfig)
    random_split: RandomSplitConfig = field(default_factory=RandomSplitConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    interpret: InterpretConfig = field(default_factory=InterpretConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    @property
    def prefix(self) -> str:
        """Turn the dataset name into a filename prefix."""
        cleaned = "".join(c if c.isalnum() else "_" for c in self.dataset.lower())
        return cleaned.strip("_") or "dataset"


def validate(cfg: RunConfig) -> None:
    """Check the invalid configuration values."""
    if cfg.evaluation_mode not in EVALUATION_MODES:
        raise ValueError(f"Invalid evaluation_mode {cfg.evaluation_mode!r}. Choose one of {sorted(EVALUATION_MODES)}.")
    unknown = [scenario for scenario in cfg.ood.scenarios if scenario not in OOD_SCENARIOS]
    if unknown:
        raise ValueError(f"Invalid ood.scenarios {unknown}. Choose from {list(OOD_SCENARIOS)}.")
    if cfg.compare_models is not None and len(cfg.compare_models) != 2:
        raise ValueError(f"compare_models must name exactly two models, got {list(cfg.compare_models)}.")


ConfigStore.instance().store(name="schema", node=RunConfig)
