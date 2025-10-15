import os
import logging


def setup_logger(logs_dir: str, log_name: str) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = os.path.join(logs_dir, f'{log_name}.txt')
    logger = logging.getLogger(log_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_filename, mode='w')
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

