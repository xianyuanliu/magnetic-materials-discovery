"""Typed run configuration, parsed and validated once from a YAML file.

Every stage used to read the raw config dict directly, mixing `cfg["x"]`
(KeyError on a missing key) with `cfg.get("x", default)` (silent fallback), and
an unrecognised key was ignored entirely — so a typo in `uq_calibration_fraction`
simply disabled it. Parsing happens here instead: unknown keys are rejected,
types are coerced once, and the rest of the codebase receives frozen dataclasses.

`RunConfig` covers settings shared across every mode; `KFoldConfig` and
`TuningConfig` carry settings shared by more than one mode but not all of
them; `HoldoutConfig`, `AblationConfig`, `OODConfig`, `UQConfig` and
`PredictConfig` carry the settings specific to one evaluation mode.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import yaml

from utils.core import DEFAULT_ALPHA
from utils.model_spec import DEFAULT_TUNE_CV_FOLDS, DEFAULT_TUNE_N_ITER

# Evaluation modes main.py can dispatch to.
EVALUATION_MODES = ("predict", "holdout", "cross_validation", "ood", "uq")

# OOD scenario families; "all" selects every one of them.
OOD_MODES = ("element", "period", "group", "cluster", "sparsex", "sparsey", "all")

# Minimum split sizes. A handful of test samples makes R^2 meaningless and a tiny train set measures nothing but the
# sample count, so both are filtered out rather than reported as OOD results.
DEFAULT_MIN_TEST = 10
DEFAULT_MIN_TRAIN = 50

# Fraction of each training set held back to calibrate conformal intervals.
DEFAULT_CALIBRATION_FRACTION = 0.25


@dataclass(frozen=True)
class OODConfig:
    """Resolved OOD settings for one run.

    Attributes:
        ood_mode: Which scenario families to build; see OOD_MODES.
        ood_k: Number of KMeans clusters for the LOCO family.
        ood_seed: Seed for the clustering and the size-matched controls.
        ood_max_splits: Cap per scenario, or None to run every split that
            passes the size filters. Targets are ordered by frequency, so a cap
            keeps precisely the largest test sets and the smallest training
            sets — a smoke-test setting, not a smaller experiment.
        elements/periods/groups: Explicit hold-out targets per family. These are
            separate keys because one shared list cannot be both element symbols
            and integer periods; see load_ood_config.
        fractions: Held-out fractions for the SparseX/SparseY families.
        size_matched_control: Score a same-size random split beside each OOD
            split, so the reported drop separates shift from lost training data.
    """

    ood_mode: str = "all"
    ood_k: int = 10
    ood_seed: int = 0
    ood_max_splits: Optional[int] = None
    elements: Optional[Tuple[str, ...]] = None
    periods: Optional[Tuple[int, ...]] = None
    groups: Optional[Tuple[int, ...]] = None
    fractions: Tuple[float, ...] = (0.1, 0.2)
    sparsex_neighbors: int = 5
    sparsey_center: str = "median"
    min_test: int = DEFAULT_MIN_TEST
    min_train: int = DEFAULT_MIN_TRAIN
    seeds: Tuple[int, ...] = (0,)
    period_strict: bool = False
    group_strict: bool = False
    size_matched_control: bool = True
    output_dir: str = "./results/ood"


@dataclass(frozen=True)
class UQConfig:
    """Resolved uncertainty-quantification settings for one run.

    Attributes:
        model: Registry key of the model the intervals are built from. It must
            be in pipeline.uq_pipeline.ENSEMBLE_STD_MODELS, checked at run time
            rather than assumed to be a Random Forest.
        alpha: Nominal miscoverage of the reported intervals.
        calibration_fraction: Fraction of each training set withheld to
            calibrate the conformal intervals.
        seeds: Repeats — one seed moves coverage by more than the effects being
            compared.
    """

    model: str = "rf"
    alpha: float = DEFAULT_ALPHA
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION
    seeds: Tuple[int, ...] = (0,)
    output_dir: str = "./results/uq"


@dataclass(frozen=True)
class PredictConfig:
    """Resolved settings for `evaluation_mode: predict`.

    Attributes:
        model: Registry key of the model to fit or load.
        model_path: Where the fitted-model bundle is saved to and loaded from.
        retrain: Refit and overwrite the bundle even if `model_path` exists.
        input_path: CSV of compositions to predict; needs the formula column.
        formulas: Compositions given inline in the config, used when
            `input_path` is unset.
        output_path: Where the prediction table is written.
    """

    model: str = "rf"
    model_path: Optional[str] = None
    retrain: bool = False
    input_path: Optional[str] = None
    formulas: Tuple[str, ...] = ()
    output_path: Optional[str] = None


@dataclass(frozen=True)
class KFoldConfig:
    """K-fold splitter settings, shared by cross_validation and by the
    in-distribution K-fold reference that OOD and UQ score every split against.
    """

    folds: int = 5
    shuffle: bool = True
    random_state: int = 0
    seeds: Tuple[int, ...] = (0,)


@dataclass(frozen=True)
class HoldoutConfig:
    """Resolved settings for `evaluation_mode: holdout`."""

    seeds: Tuple[int, ...] = (0,)
    train_size: float = 0.8


@dataclass(frozen=True)
class TuningConfig:
    """Hyperparameter-search budget, shared by cross_validation and OOD.

    The search re-runs inside every outer fold, so its cost is
    (outer folds x seeds x tunable models x n_iter x cv_folds) model fits —
    keep cv_folds and n_iter well below the outer fold count.
    """

    enabled: bool = False
    cv_folds: int = DEFAULT_TUNE_CV_FOLDS
    n_iter: int = DEFAULT_TUNE_N_ITER


@dataclass(frozen=True)
class AblationConfig:
    """Interpretability-figure settings, read only by the holdout pipeline."""

    enabled: bool = False
    interpret_model: Optional[str] = None
    case_study_models: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    """One fully resolved run, as parsed from a YAML config file.

    Attributes:
        feature_columns: Feature columns to use, named explicitly. Unset falls
            back to "every column that is not the target or the formula",
            which promotes any stray id or metadata column into a model input.
        compare_models: The two models the paired comparison reports on.
    """

    dataset: str
    evaluation_mode: str
    pt_path: str
    mm_path: str

    target_column: str = "saturation magnetization"
    formula_column: str = "chemical formula"
    feature_columns: Optional[Tuple[str, ...]] = None

    dataset_path: Optional[str] = None
    train_dataset_path: Optional[str] = None
    test_dataset_path: Optional[str] = None

    models: Tuple[str, ...] = ()
    compare_models: Optional[Tuple[str, str]] = None

    enable_data_visualization: bool = False

    model_random_state: int = 0
    plots_output_dir: str = "./plots"

    kfold: KFoldConfig = field(default_factory=KFoldConfig)
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    uq: UQConfig = field(default_factory=UQConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    @property
    def prefix(self) -> str:
        """Filename prefix for this run's figures, derived from the dataset name.

        Previously a hard-coded novamag/mp lookup that raised on anything else,
        which meant a new dataset could not run at all until main.py was edited.
        """
        cleaned = "".join(c if c.isalnum() else "_" for c in self.dataset.lower())
        return cleaned.strip("_") or "dataset"


# Every key a config file may contain. Anything else is a typo, and is reported as one rather than silently ignored.
_TOP_LEVEL_KEYS = frozenset({
    "dataset", "evaluation_mode", "pt_path", "mm_path",
    "target_column", "formula_column", "feature_columns",
    "dataset_path", "train_dataset_path", "test_dataset_path",
    "models", "compare_models", "interpret_model", "case_study_models",
    "enable_data_visualization", "enable_hyperparameter_tuning", "enable_ablation_study",
    "cv_folds", "cv_shuffle", "cv_random_state", "cv_seeds",
    "holdout_seeds", "holdout_train_size",
    "tune_cv_folds", "tune_n_iter", "random_state", "plots_output_dir",
})

_OOD_KEYS = frozenset({
    "ood_mode", "ood_k", "ood_seed", "ood_max_splits",
    "ood_elements", "ood_periods", "ood_groups",
    "ood_fractions", "sparsex_neighbors", "sparsey_center",
    "ood_min_test", "ood_min_train", "ood_period_strict", "ood_group_strict",
    "ood_size_matched_control", "ood_output_dir",
})

_UQ_KEYS = frozenset({"uq_model", "uq_alpha", "uq_calibration_fraction", "uq_seeds", "uq_output_dir"})

_PREDICT_KEYS = frozenset({
    "predict_model", "predict_model_path", "predict_retrain",
    "predict_input_path", "predict_formulas", "predict_output_path",
})

KNOWN_KEYS = _TOP_LEVEL_KEYS | _OOD_KEYS | _UQ_KEYS | _PREDICT_KEYS


def _as_tuple(value, cast=None) -> Optional[Tuple]:
    """Coerce a scalar or sequence to a tuple, passing None through."""
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else [value]
    return tuple(cast(v) for v in items) if cast else tuple(items)


def _check_known_keys(raw: Mapping[str, Any]) -> None:
    """Raise if the config carries a key no stage reads.

    Raises:
        ValueError: Listing the unknown keys and the closest known ones.
    """
    unknown = sorted(set(raw) - KNOWN_KEYS)
    if not unknown:
        return

    import difflib

    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(KNOWN_KEYS), n=1)
        hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise ValueError("Unknown config key(s): " + ", ".join(hints))


def load_ood_config(raw: Mapping[str, Any], default_seed: int) -> OODConfig:
    """Build an OODConfig from the raw mapping, filling in defaults.

    Raises:
        ValueError: If `ood_mode` is unsupported.
    """
    mode = str(raw.get("ood_mode", "all")).lower()
    if mode not in OOD_MODES:
        raise ValueError(f"Invalid ood_mode {mode!r}. Supported: {sorted(OOD_MODES)}")

    elements = _as_tuple(raw.get("ood_elements"), str)
    periods = _as_tuple(raw.get("ood_periods"), int)
    groups = _as_tuple(raw.get("ood_groups"), int)

    max_splits = raw.get("ood_max_splits")
    seeds = _as_tuple(raw.get("cv_seeds"), int) or (int(raw.get("ood_seed", default_seed)),)

    return OODConfig(
        ood_mode=mode,
        ood_k=int(raw.get("ood_k", 10)),
        ood_seed=int(raw.get("ood_seed", default_seed)),
        ood_max_splits=None if max_splits is None else int(max_splits),
        elements=elements,
        periods=periods,
        groups=groups,
        fractions=_as_tuple(raw.get("ood_fractions", [0.1, 0.2]), float),
        sparsex_neighbors=int(raw.get("sparsex_neighbors", 5)),
        sparsey_center=str(raw.get("sparsey_center", "median")),
        min_test=int(raw.get("ood_min_test", DEFAULT_MIN_TEST)),
        min_train=int(raw.get("ood_min_train", DEFAULT_MIN_TRAIN)),
        seeds=seeds,
        period_strict=bool(raw.get("ood_period_strict", False)),
        group_strict=bool(raw.get("ood_group_strict", False)),
        size_matched_control=bool(raw.get("ood_size_matched_control", True)),
        output_dir=str(raw.get("ood_output_dir", "./results/ood")),
    )


def _load_uq_config(raw: Mapping[str, Any], default_seed: int) -> UQConfig:
    """Build a UQConfig from the raw mapping, filling in defaults."""
    seeds = (
        _as_tuple(raw.get("uq_seeds"), int)
        or _as_tuple(raw.get("cv_seeds"), int)
        or (default_seed,)
    )
    return UQConfig(
        model=str(raw.get("uq_model", "rf")),
        alpha=float(raw.get("uq_alpha", DEFAULT_ALPHA)),
        calibration_fraction=float(raw.get("uq_calibration_fraction", DEFAULT_CALIBRATION_FRACTION)),
        seeds=seeds,
        output_dir=str(raw.get("uq_output_dir", "./results/uq")),
    )


def _load_predict_config(raw: Mapping[str, Any], models: Sequence[str]) -> PredictConfig:
    """Build a PredictConfig, defaulting the model to the first one configured."""
    default_model = str(raw.get("predict_model", models[0] if models else "rf"))
    return PredictConfig(
        model=default_model,
        model_path=raw.get("predict_model_path"),
        retrain=bool(raw.get("predict_retrain", False)),
        input_path=raw.get("predict_input_path"),
        formulas=_as_tuple(raw.get("predict_formulas", []), str) or (),
        output_path=raw.get("predict_output_path"),
    )


def _validate(cfg: RunConfig) -> None:
    """Check cross-field constraints that a single key cannot express.

    Raises:
        ValueError: On an unsupported mode, a missing dataset path for the
            chosen mode, or a `compare_models` pair that is not a pair.
    """
    if cfg.evaluation_mode not in EVALUATION_MODES:
        raise ValueError(
            f"Invalid evaluation_mode {cfg.evaluation_mode!r}. "
            f"Choose one of {sorted(EVALUATION_MODES)}."
        )

    if cfg.evaluation_mode == "ood":
        if not cfg.train_dataset_path:
            raise ValueError("OOD mode requires train_dataset_path in the config.")
    elif not cfg.dataset_path:
        raise ValueError(f"dataset_path is required for evaluation_mode: {cfg.evaluation_mode}.")

    if cfg.compare_models is not None and len(cfg.compare_models) != 2:
        raise ValueError(f"compare_models must name exactly two models, got {list(cfg.compare_models)}.")

    if cfg.feature_columns is not None and not cfg.feature_columns:
        raise ValueError("feature_columns was given but is empty.")

    if not 0.0 < cfg.holdout.train_size < 1.0:
        raise ValueError(f"holdout_train_size must be in (0, 1), got {cfg.holdout.train_size}.")


def parse_run_config(raw: Mapping[str, Any]) -> RunConfig:
    """Turn a raw config mapping into a validated RunConfig.

    Args:
        raw: The parsed YAML mapping.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: On an unknown key or a failed cross-field check.
    """
    _check_known_keys(raw)

    models = _as_tuple(raw.get("models", []), str) or ()
    cv_random_state = int(raw.get("cv_random_state", 0))

    cfg = RunConfig(
        dataset=str(raw["dataset"]),
        evaluation_mode=str(raw["evaluation_mode"]).lower(),
        pt_path=str(raw["pt_path"]),
        mm_path=str(raw["mm_path"]),
        target_column=str(raw.get("target_column", "saturation magnetization")),
        formula_column=str(raw.get("formula_column", "chemical formula")),
        feature_columns=_as_tuple(raw.get("feature_columns"), str),
        dataset_path=raw.get("dataset_path"),
        train_dataset_path=raw.get("train_dataset_path"),
        test_dataset_path=raw.get("test_dataset_path"),
        models=models,
        compare_models=_as_tuple(raw.get("compare_models"), str),
        enable_data_visualization=bool(raw.get("enable_data_visualization", False)),
        model_random_state=int(raw.get("random_state", 0)),
        plots_output_dir=str(raw.get("plots_output_dir", "./plots")),
        ablation=AblationConfig(
            enabled=bool(raw.get("enable_ablation_study", False)),
            interpret_model=raw.get("interpret_model"),
            case_study_models=_as_tuple(raw.get("case_study_models", []), str) or (),
        ),
        kfold=KFoldConfig(
            folds=int(raw.get("cv_folds", 5)),
            shuffle=bool(raw.get("cv_shuffle", True)),
            random_state=cv_random_state,
            seeds=_as_tuple(raw.get("cv_seeds"), int) or (cv_random_state,),
        ),
        holdout=HoldoutConfig(
            seeds=_as_tuple(raw.get("holdout_seeds"), int) or (cv_random_state,),
            train_size=float(raw.get("holdout_train_size", 0.8)),
        ),
        tuning=TuningConfig(
            enabled=bool(raw.get("enable_hyperparameter_tuning", False)),
            cv_folds=int(raw.get("tune_cv_folds", DEFAULT_TUNE_CV_FOLDS)),
            n_iter=int(raw.get("tune_n_iter", DEFAULT_TUNE_N_ITER)),
        ),
        ood=load_ood_config(raw, default_seed=cv_random_state),
        uq=_load_uq_config(raw, default_seed=cv_random_state),
        predict=_load_predict_config(raw, models),
    )

    _validate(cfg)
    return cfg


def load_run_config(path: str) -> RunConfig:
    """Read a YAML config file and parse it into a RunConfig.

    Raises:
        ValueError: If the file is empty or does not hold a mapping.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return parse_run_config(raw)
