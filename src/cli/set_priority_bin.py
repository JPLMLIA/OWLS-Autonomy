"""
Sets the priority bin for an ASDP
"""
import os.path as op
import argparse
import logging

from jewel.asdpdb import ASDPDB
from utils import logger as OWLSlogger

def set_priority_bin(dbfile, asdpid, priority):
    logger.info('Loading ASDPDB...')
    asdpdb = ASDPDB(dbfile)
    asdpdb.set_priority_bin(asdpid, priority)
    logger.info(f'Set priority bin of ASDP {asdpid} to {priority}')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('dbfile',               help='path to the ASDP DB CSV file')

    parser.add_argument('asdpid',               type=int, 
                                                help='integer ASDP identifier')

    parser.add_argument('priority',             type=int,
                                                help='a new downlink priority bin for the ASDP')

    parser.add_argument('--log_name',           default="set_priority_bin.log",
                                                help="Filename for the pipeline log. Default is set_priority_bin.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    OWLSlogger.setup_logger(args.log_name, args.log_folder)
    global logger
    logger = logging.getLogger()
    
    kwargs = vars(args)
    kwargs.pop('log_name', None)
    kwargs.pop('log_folder', None)
    set_priority_bin(**kwargs)
