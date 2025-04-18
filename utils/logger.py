
import logging

def get_logger():
    logger = logging.getLogger("RAGAppLogger")
    if not logger.handlers:
        handler = logging.FileHandler("app.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
