import os
import os.path as op
import argparse
import yaml
import logging

from jewel.asdpdb import (
    ASDPDB,
    load_asdp_metadata_by_bin_and_type,
    save_asdp_ordering,
)
from jewel.prioritize import load_prioritizer
from utils            import logger as OWLSlogger

def invoke_jewel(dbfile, outputfile, config):

    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    logger.info('Loading ASDPDB...')
    asdpdb = ASDPDB(dbfile)
    asdpdb_data = load_asdp_metadata_by_bin_and_type(asdpdb)

    logger.info('Loading prioritizer...')
    prioritize = load_prioritizer(cfg)

    logger.info('Prioritizing...')
    ordering = prioritize(asdpdb_data)

    logger.info('Saving prioritized list...')
    save_asdp_ordering(outputfile, ordering)

    logger.info('Done.')


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('dbfile',               help='path to the ASDP DB CSV file')

    parser.add_argument('outputfile',           help='path to output (ASDP ordering CSV file)')

    parser.add_argument('-c', '--config',       default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'jewel_default.yml'),
                                                help='Prioritization configuration file (YML). Defaults to cli/configs/jewel_default.yml')

    parser.add_argument('--log_name',           default="JEWEL.log",
                                                help="Filename for the pipeline log. Default is JEWEL.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    OWLSlogger.setup_logger(args.log_name, args.log_folder)
    global logger
    logger = logging.getLogger()
    
    kwargs = vars(args)
    kwargs.pop('log_name', None)
    kwargs.pop('log_folder', None)
    invoke_jewel(**kwargs)
