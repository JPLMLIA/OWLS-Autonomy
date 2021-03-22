import logging
import pathlib
import os
import yaml
import inspect

from logging.handlers import RotatingFileHandler

class ColorFormatter(logging.Formatter):
    """
    Logging formatter that adds color to warning and error messages for emphasis
    """

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = '%(asctime)s | %(levelname)-7s | %(module)-15s | %(message)s'
    datefmt = "%m-%d %H:%M.%S"

    FMTS = {
        logging.DEBUG: (fmt, datefmt),
        logging.INFO: (fmt, datefmt),
        logging.WARNING: (yellow+fmt+reset, datefmt),
        logging.ERROR: (red+fmt+reset, datefmt),
        logging.CRITICAL: (bold_red+fmt+reset, datefmt)
    }

    def format(self, record):
        log_fmt = self.FMTS.get(record.levelno)
        formatter = logging.Formatter(*log_fmt)
        return formatter.format(record)

def get_logger(script_name=None, logger_name=None,
        DEFAULT_WRITE_DIR='output', setup=False):
    """
    Set up logger and return a Logger instance

    Parameters
    ----------
    script_name: str or None
        name of the script used to set the logger output file name; if None,
        inspection is used to get the caller script name
    logger_name: str or None
        name used to instantiate the logger; if None, inspection is used to get
        the caller module name
    DEFAULT_WRITE_DIR: str (default: 'output')
        output directory where log file will be written
    setup: bool (default: False)
        also perform global logger setup before returning a Logger instance.
        This is intended to be used at the top level of your processing chain 
        (usually the CLI script).

    Returns
    -------
    logger: logging.Logger
        a logger instance with the `logger_name`
    """
    if script_name is None or logger_name is None:
        # Use inspection to get the caller's script/module name
        frames = inspect.stack()
        assert len(frames) >= 1
        caller_frame = frames[1]
        if script_name is None:
            filename = caller_frame[0].f_code.co_filename
            script_name = os.path.splitext(os.path.basename(filename))[0]
        if logger_name is None:
            logger_name = inspect.getmodule(caller_frame[0]).__name__

    if setup:
        setup_logger(script_name, DEFAULT_WRITE_DIR)

    return logging.getLogger(logger_name)

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
                        format='%(asctime)s | %(levelname)-7s | %(module)-15s | %(message)s',
                        datefmt='%m-%d %H:%M.%S',
                        filename=os.path.join(DEFAULT_WRITE_DIR, script_name),
                        filemode='a')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)

    # set a format which is simpler for console use
    #formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(module)-15s | %(message)s', "%Y-%m-%d %H:%M.%S")

    # tell the handler to use this format
    #console.setFormatter(formatter)  # Simple formatter
    console.setFormatter(ColorFormatter())  # Color formatter

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
