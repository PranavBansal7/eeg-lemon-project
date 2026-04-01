from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.pkl"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "processed" / "feature_columns.json"

TARGET_NAMES = [
    "working_memory",
    "attention",
    "executive_function",
    "intelligence",
]

CONDITIONS = ("eo", "ec")
ANCHOR_CHANNELS = ("cz", "f3", "f4", "fz", "o1")
FRONTAL_ANCHORS = ("f3", "f4", "fz")
CENTRAL_ANCHORS = ("cz",)
POSTERIOR_ANCHORS = ("o1",)

BANDS = (
    "delta",
    "theta",
    "low_alpha",
    "high_alpha",
    "alpha",
    "low_beta",
    "beta",
    "high_beta",
)
