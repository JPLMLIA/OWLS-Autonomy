import logging
import pathlib
import os
import yaml

from logging.handlers import RotatingFileHandler

def setup_logger(script_name, DEFAULT_WRITE_DIR):
    """ Sets up logger.
    """

    LOG_LEVEL = logging.INFO

    # Logger will not be set up twice.
    if logging.getLogger('').handlers:
        return

    # make directory if necessary
    pathlib.Path(DEFAULT_WRITE_DIR).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s | %(levelname)-7s | %(message)s',
                        datefmt='%m-%d %H:%M.%S',
                        filename=os.path.join(DEFAULT_WRITE_DIR, script_name + '.log'),
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', "%Y-%m-%d %H:%M.%S")

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    