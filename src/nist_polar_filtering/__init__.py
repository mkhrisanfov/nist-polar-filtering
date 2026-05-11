from pathlib import Path

import seaborn as sns
from numpy import array, geomspace

SEED_START = 42
SEED_END = 46
RADIUS = 3
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data/output"
PROCESSED_DIR = BASE_DIR / "data/processed"
POLAR_CLASSES = [
    "semi-std-polar",
    "Standard polar",
    "std-polar",
    "Standard semi-polar",
    "SP",
]
SNP_COLUMN_CLASSES = ["Standard non-polar", "SNP", "std-non-polar"]
SSNP_COLUMN_CLASSES = [
    "SSNP",
    "Semi-standard non-polar",
    "semi-std-non-polar",
    "Semi-standard-non-polar",
    "Semi-standatd non-polar",
]
MODEL_NAMES = ["cnn", "gnn"]
SEED = 43
DEFAULT_DATA_OFFSET = 1.5
DEFAULT_DATA_OFFSETS = array((0, 1.5, 10))
DEFAULT_THRESHOLD = 5.0
DEFAULT_THRESHOLDS = geomspace(0.1, 100, 15)
TOL = 1e-8
colorblind = sns.color_palette("colorblind")
new_order = [colorblind[i] for i in [0, 3, 7, 2, 5, 4, 6, 8, 1, 9]]
sns.palettes.SEABORN_PALETTES["colorblind_reordered"] = new_order
PALETTE = sns.color_palette("colorblind_reordered")
HATCH_COEFF = 5
HATCHES = [
    "/" * HATCH_COEFF,
    "\\" * HATCH_COEFF,
    "x" * HATCH_COEFF,
    "o" * HATCH_COEFF,
    "O" * HATCH_COEFF,
    "." * HATCH_COEFF,
    "*" * HATCH_COEFF,
    "|" * HATCH_COEFF,
    "-" * HATCH_COEFF,
    "+" * HATCH_COEFF,
]
HATCH_DICT = dict(zip(new_order, HATCHES))

__all__ = [
    "BASE_DIR",
    "OUTPUT_DIR",
    "PROCESSED_DIR",
    "SEED",
    "SEED_START",
    "SEED_END",
    "RADIUS",
    "MODEL_NAMES",
    "POLAR_CLASSES",
    "SNP_COLUMN_CLASSES",
    "SSNP_COLUMN_CLASSES",
    "DEFAULT_DATA_OFFSET",
    "DEFAULT_DATA_OFFSETS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "TOL",
    "PALETTE",
    "HATCHES",
    "HATCH_DICT",
]
