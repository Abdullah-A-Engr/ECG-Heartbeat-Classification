"""Load the MIT-BIH heartbeat CSV files (187 signal values + label)."""
from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def _read_csv_or_zip(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise FileNotFoundError(f"No CSV found in {path}")
            with zf.open(names[0]) as f:
                return pd.read_csv(f, header=None)
    return pd.read_csv(path, header=None)

def load_data(train_path=None, test_path=None):
    train_path = Path(train_path) if train_path else DATA / "mitbih_train.csv.zip"
    test_path = Path(test_path) if test_path else DATA / "mitbih_test.csv.zip"
    return _read_csv_or_zip(train_path), _read_csv_or_zip(test_path)
