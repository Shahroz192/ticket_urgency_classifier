from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Load configuration from YAML
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / config["data_dir"]
RAW_DATA_DIR = PROJ_ROOT / config["raw_data_dir"]
INTERIM_DATA_DIR = PROJ_ROOT / config["interim_data_dir"]
PROCESSED_DATA_DIR = PROJ_ROOT / config["processed_data_dir"]
EXTERNAL_DATA_DIR = PROJ_ROOT / config["external_data_dir"]

MODELS_DIR = PROJ_ROOT / config["models_dir"]

REPORTS_DIR = PROJ_ROOT / config["reports_dir"]
FIGURES_DIR = PROJ_ROOT / config["figures_dir"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
