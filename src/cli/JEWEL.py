import os
import os.path as op
import argparse
import yaml
import logging
import timeit

from jewel.asdpdb import (
    ASDPDB,
    load_asdp_metadata_by_bin_and_type,
    save_asdp_ordering,
)
from jewel.prioritize import load_prioritizer
from utils            import logger as OWLSlogger
from utils.memory_tracker.plotter  import Plotter, watcher

def invoke_jewel(dbfile, outputfile, config):

    st = timeit.default_timer()

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

    et = timeit.default_timer()
    logging.info(f"JEWEL run completed in {et-st:.2f} s")

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

    # setup the plotter
    pltt = Plotter(save_to=op.join(args.log_folder, "JEWEL_memory.mp4"))
    globalQ = pltt.get_queues('JEWEL.py')

    # Set up the watcher arguments
    watch = {'JEWEL.py': {'queue': globalQ.graph, 'pid': os.getpid()}}

    # Start watcher then the plotter
    watcher(watch)
    pltt.start()
    
    kwargs = vars(args)
    kwargs.pop('log_name', None)
    kwargs.pop('log_folder', None)
    invoke_jewel(**kwargs)

    try:
        ram_mean, ram_max = pltt.stop()
        logging.info(f'Average RAM:{ram_mean:.2f}GB, Max RAM:{ram_max:.2f}GB')
    except:
        logging.error("Memory tracker failed to shut down correctly.")

