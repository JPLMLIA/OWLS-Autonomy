"""
Command line interface to update the ASDP DB for JEWEL and tracking
product downlink status.
"""
import os.path as op
import argparse
import logging

from tqdm import tqdm
from glob import glob

from fsw.JEWEL.asdpdb     import ASDPDB, compile_asdpdb_entry
from utils.dir_helper     import get_unique_file_by_suffix
from utils                import logger as OWLSlogger

def update_asdp_db(rootdir_glob, dbfile, log_folder, log_name):

    OWLSlogger.setup_logger(log_name, log_folder)
    logger = logging.getLogger()
    
    asdp_db = ASDPDB(dbfile)

    # Check each directory for a manifest file
    experiments = []
    rootdirs = list(glob(rootdir_glob))
    logger.info(f'{len(rootdirs)} experiment directories provided as input.')

    for rootdir in tqdm(rootdirs, desc='Checking for manifest files'):
        manifest_file = get_unique_file_by_suffix(
            rootdir, 'manifest.json', logger=logger
        )
        if manifest_file is not None:
            experiments.append((rootdir, manifest_file))

    logger.info(f'Found {len(experiments)} experiment directories')

    # Filter new experiment directories
    new_experiments = [
        (e, m) for e, m in experiments
        if not asdp_db.entry_exists(e)
    ]
    logger.info(f'Found {len(new_experiments)} new experiment directories')

    if len(new_experiments) > 0:
        new_entries = [
            compile_asdpdb_entry(e, m)
            for e, m in new_experiments
        ]
        new_good_entries = [e for e in new_entries if e is not None]
        logger.info(f'Prepared {len(new_good_entries)} entries for ASDPDB')

        if len(new_good_entries) > 0:
            inserted = asdp_db.add_entries(new_good_entries)
            logger.info(f'Updated ASDP DB with {len(inserted)} entries')

    # Shut down all open loggers so they do not interfere with future runs in the same session
    for x in range(0, len(logging.getLogger().handlers)):
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--rootdirs',           help='Glob of experiment root directories')

    parser.add_argument('--dbfile',             help='path to db CSV file (will be created if it does not exist)')

    parser.add_argument('--log_name',           default="update_asdp_db.log",
                                                help="Filename for the pipeline log. Default is update_asdp_db.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    update_asdp_db(args.rootdirs, args.dbfile, args.log_folder, args.log_name)

if __name__ == "__main__":
    main()
