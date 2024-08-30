import logging
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logger():
    # Create the logs directory if it doesn't exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Set up logging
    log_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(lineno)d : %(message)s'
    )

    # Create a rotating file handler
    log_file_handler = TimedRotatingFileHandler(
        'logs/record.log',
        when='midnight',
        interval=1,
        backupCount=15,
        encoding='utf-8',
        delay=False,
        utc=True,
    )
    log_file_handler.setFormatter(log_formatter)

    # Get the root logger
    logger = logging.getLogger()

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Configure the root logger
    logger.setLevel(logging.INFO)
    logger.addHandler(log_file_handler)

    return logging.getLogger(__name__)
