import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers
    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        maxBytes=5_000_000,
        backupCount=3
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
