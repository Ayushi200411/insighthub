import logging
import os
from datetime import datetime

LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

logger = logging.getLogger("insighthub")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_FOLDER, "agent_trace.log"), encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_step(step_name, details=""):
    logger.info(f"{step_name} | {details}")