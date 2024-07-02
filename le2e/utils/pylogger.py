import logging
from lightning.pytorch.utilities import rank_zero_only


MESSAGE_FORMAT = (
    "%(levelname)s %(asctime)s:  %(message)s"
)
DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
LG_FORMATTER = logging.Formatter(MESSAGE_FORMAT, datefmt=DATE_FORMAT)


def get_script_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    stderr_h = logging.StreamHandler()
    stderr_h.setFormatter(LG_FORMATTER)
    stderr_h.setLevel(logging.INFO)
    logger.addHandler(stderr_h)
    logger.setLevel(logging.DEBUG)
    return logger


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    :param name: The name of the logger, defaults to ``__name__``.

    :return: A logger object.
    """
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

