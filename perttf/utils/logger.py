import logging
import sys
def create_logger(id=0):

    logger = logging.getLogger(f"pertTF_worker_{id}")
# check if logger has been initialized
    if not logger.hasHandlers() or len(logger.handlers) == 0:
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger