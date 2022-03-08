"""
Sets the priority bin for an ASDP
"""
import os.path as op
import argparse
import logging

from fsw.JEWEL.asdpdb import ASDPDB
from utils import logger as OWLSlogger

def set_priority_bin(dbfile, asdpid, priority, log_folder, log_name):

    OWLSlogger.setup_logger(log_name, log_folder)
    logger = logging.getLogger()

    logger.info('Loading ASDPDB...')

    asdpdb = ASDPDB(dbfile)
    asdpdb.set_priority_bin(asdpid, priority)

    logger.info(f'Set priority bin of ASDP {asdpid} to {priority}')

    # Shut down all open loggers so they do not interfere with future runs in the same session
    for x in range(0, len(logging.getLogger().handlers)):
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dbfile',             help='path to the ASDP DB CSV file')

    parser.add_argument('--asdpid',             type=int, 
                                                help='integer ASDP identifier')

    parser.add_argument('--priority',           type=int,
                                                help='a new downlink priority bin for the ASDP')

    parser.add_argument('--log_name',           default="set_priority_bin.log",
                                                help="Filename for the pipeline log. Default is set_priority_bin.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    set_priority_bin(args.dbfile, args.asdpid, args.priority, args.log_folder, args.log_name)

if __name__ == "__main__":
    main()