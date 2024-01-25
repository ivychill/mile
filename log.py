
import logging.handlers

def get_logger():
    logger = logging.getLogger('voice')
    return logger

logger = get_logger()

def set_logger(logger, log_file=None):
    MAX_LOG_SIZE = 2560000
    LOG_BACKUP_NUM = 4000
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    if log_file is not None:
        ff = logging.Formatter(
            '%(asctime)s %(process)d %(processName)s %(filename)s %(lineno)d %(levelname)s %(message)s')
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.info("logger set up")