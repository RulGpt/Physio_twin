from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = PROJECT_ROOT / "database"

DATA_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)