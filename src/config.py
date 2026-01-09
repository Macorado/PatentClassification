from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("figures")
    metrics_dir: Path = Path("metrics")

    office_actions: Path = data_dir / "office_actions.pkl"
    rejections: Path = data_dir / "rejections.pkl"
    citations: Path = data_dir / "citations.pkl"

@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    test_size: float = 0.2

    target_col: str = "allowed_claims"
    drop_cols: tuple = ("app_id", "first_mail_date", "last_mail_date")
    cat_cols: tuple = ("art_unit", "uspc_class", "uspc_subclass")

    baseline_subsample_n: int = 200_000 