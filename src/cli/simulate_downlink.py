"""
Simulates the downlink
"""
import os.path as op
import argparse
import logging

from jewel.asdpdb import (
    ASDPDB, load_asdp_ordering, DownlinkStatus,
)
from utils        import logger as OWLSlogger

def simulate_downlink(dbfile, orderfile, datavolume):

    logger.info('Loading ASDPDB...')
    asdpdb = ASDPDB(dbfile)
    ordering = load_asdp_ordering(orderfile)

    remaining_volume = datavolume
    logger.info(f'Downlink data volume: {remaining_volume}')

    for o in ordering:
        # Load entry
        asdp_id = o['asdp_id']
        entry = asdpdb.get_entry_by_id(asdp_id)

        # Continue if entry is untransmitted
        if entry['downlink_status'] == DownlinkStatus.UNTRANSMITTED:

            # Continue if ASDP fits in remaining data volume
            size = int(entry['asdp_size_bytes'])
            if size <= remaining_volume:

                # Perform downlink
                logger.info(f'Downlinking ASDP {asdp_id}')
                asdpdb.set_downlink_status(
                    asdp_id, DownlinkStatus.TRANSMITTED
                )
                remaining_volume -= size
                logger.info(f'Remaining data volume: {remaining_volume}')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('dbfile',               help='path to the ASDP DB CSV file')

    parser.add_argument('orderfile',            help='path to the ASDP ordering CSV file')

    parser.add_argument('datavolume',           type=int,
                                                help='downlink data volume in bytes')

    parser.add_argument('--log_name',           default="simulate_downlink.log",
                                                help="Filename for the pipeline log. Default is simulate_downlink.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    OWLSlogger.setup_logger(args.log_name, args.log_folder)
    global logger
    logger = logging.getLogger()
    
    kwargs = vars(args)
    kwargs.pop('log_name', None)
    kwargs.pop('log_folder', None)
    simulate_downlink(**kwargs)
