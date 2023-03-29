"""
Simulates the downlink
"""
import os
import os.path as op
import argparse
import logging
from datetime import datetime
from shutil import copy, copytree

from fsw.JEWEL.asdpdb import (
    ASDPDB, load_asdp_ordering, DownlinkStatus,
)
from utils        import logger as OWLSlogger
from utils.manifest import AsdpManifest

def copy_asdp(entry, destination):
    manifestfile = entry['manifest_file']
    manifest = AsdpManifest.load(manifestfile)

    # Get paths and relative directories
    rel_paths = [
        e['relative_path'] for e in manifest.entries
    ]
    paths = [
        manifest.relocated_path(entry['experiment_dir'], e)
        for e in manifest.entries
    ]
    rel_dirs = set(map(op.dirname, rel_paths))
    rel_dirs.add('')

    # Make destination subdirectories
    for rel_dir in rel_dirs:
        dst_dir = op.join(destination, rel_dir)
        if not op.exists(dst_dir):
            os.makedirs(dst_dir)

    # Copy manifest
    copy(manifestfile, destination)

    # Copy manifest entries
    for path, rel_path in zip(paths, rel_paths):
        if not op.exists(path):
            logging.warning(f'"{path}" does not exist; skipping')
            continue

        dst = op.join(destination, rel_path)
        if op.isdir(path):
            copytree(path, dst)
        else:
            copy(path, dst)

def simulate_downlink(dbfile, orderfile, datavolume, downlinkdir, log_folder, log_name):

    OWLSlogger.setup_logger(log_name, log_folder)
    logger = logging.getLogger()

    if downlinkdir is not None:
        if not op.exists(downlinkdir):
            raise ValueError(
                'Downlink directory "%s" does not exist.' % downlinkdir
            )

        now = datetime.now()
        nowstr = now.strftime('%Y%m%dT%H%M%S')
        sessiondir = op.join(downlinkdir, nowstr)
        if op.exists(sessiondir):
            raise ValueError('Downlink session "%s" already exists.' % nowstr)

        os.mkdir(sessiondir)

    else:
        sessiondir = None
        logger.info('Simulating downink without copying files')

    logger.info('Loading ASDPDB...')
    asdpdb = ASDPDB(dbfile)
    ordering = load_asdp_ordering(orderfile)

    # Copy ASDP DB and ordering files to downlink session
    if sessiondir is not None:
        asdpdb_dest = op.join(sessiondir, 'asdpdb.csv')
        order_dest = op.join(sessiondir, 'ordering.csv')
        copy(dbfile, asdpdb_dest)
        copy(orderfile, order_dest)

    remaining_volume = datavolume if datavolume >= 0 else float('inf')
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
                if sessiondir is not None:
                    asdpdir = op.join(sessiondir, 'asdp%09d' % int(asdp_id))
                    copy_asdp(entry, asdpdir)

                asdpdb.set_downlink_status(
                    asdp_id, DownlinkStatus.TRANSMITTED
                )
                remaining_volume -= size
                logger.info(f'Remaining data volume: {remaining_volume}')

    # Shut down all open loggers so they do not interfere with future runs in the same session
    for x in range(0, len(logging.getLogger().handlers)):
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dbfile',             help='path to the ASDP DB CSV file')

    parser.add_argument('--orderfile',          help='path to the ASDP ordering CSV file')

    parser.add_argument('--datavolume',         type=int,
                                                help='downlink data volume in bytes (negative values means no limit)')

    parser.add_argument('-d', '--downlinkdir',  default=None,
                                                help='copy ASDP files to this directory')

    parser.add_argument('--log_name',           default="simulate_downlink.log",
                                                help="Filename for the pipeline log. Default is simulate_downlink.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()
    
    simulate_downlink(args.dbfile, args.orderfile, args.datavolume, args.downlinkdir, args.log_folder, args.log_name)

if __name__ == "__main__":
    main()
