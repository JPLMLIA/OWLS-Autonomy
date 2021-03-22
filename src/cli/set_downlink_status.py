"""
Sets the downlink status for an ASDP
"""
import os.path as op
import argparse
import logging

from jewel.asdpdb import (
    ASDPDB, DownlinkStatus,
)
from utils import logger as OWLSlogger

def set_downlink_status(dbfile, asdpid, status):
    logger.info('Loading ASDPDB...')
    asdpdb = ASDPDB(dbfile)
    asdpdb.set_downlink_status(asdpid, status)
    logger.info(f'Set status of ASDP {asdpid} to {status}')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('dbfile',               help='path to the ASDP DB CSV file')

    parser.add_argument('asdpid',               type=int, 
                                                help='integer ASDP identifier')

    parser.add_argument('status',               choices=[DownlinkStatus.UNTRANSMITTED, DownlinkStatus.TRANSMITTED],
                                                help='a new downlink status for the ASDP')

    parser.add_argument('--log_name',           default="set_downlink_status.log",
                                                help="Filename for the pipeline log. Default is set_downlink_status.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    OWLSlogger.setup_logger(args.log_name, args.log_folder)
    global logger
    logger = logging.getLogger()
    
    kwargs = vars(args)
    kwargs.pop('log_name', None)
    kwargs.pop('log_folder', None)
    set_downlink_status(**kwargs)
