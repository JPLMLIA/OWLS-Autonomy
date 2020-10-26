'''
Command line interface to the ACME data processing pipeline
'''
import sys
import argparse
import glob
import os
import os.path as op
import time
import logging
import warnings
import multiprocessing

from pathlib import Path
from sys     import platform
from tqdm    import tqdm

sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))

from utils.memory_tracker.plotter    import Plotter, watcher
from acme_cems.lib.analyzer          import analyse_experiment
from utils                           import logger

warnings.filterwarnings('ignore')

logger.setup_logger(os.path.basename(__file__).rstrip(".py"), "output")
logger = logging.getLogger(__name__)


def make_dirs(outdir):
    '''Create directory in path outdir '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)


def create_result_dirs(args, basedir, label):
    '''Creates directories where to save results depending on running options'''

    if (args['field_mode']):
        outdir = os.path.join(basedir, label)
        make_dirs(os.path.join(outdir, "Mugshots"))
    else:
        if (args['knowntraces']):
            outdir = os.path.join(basedir, label, 'Known_Masses')
        else:
            outdir = os.path.join(basedir, label, 'Unknown_Masses')

        make_dirs(outdir)
        make_dirs(os.path.join(outdir, "Heat_Maps"))
        make_dirs(os.path.join(outdir, "Mugshots"))
        if (args['debug_plots']):
            make_dirs(os.path.join(outdir, "Debug_Plots"))

        if (args['saveheatmapdata']) and not (args['knowntraces']):
            make_dirs(os.path.join(outdir, "Data_Files"))

        if not (args['noplots']):
            make_dirs(os.path.join(outdir, "Time_Trace"))
            make_dirs(os.path.join(outdir, "Mass_Spectra"))

    return outdir

def bulk_reprocess(args):

    if args['reprocess_version'] is None:
        logging.warning('Reproocessing version must be set for bulk reprocessing.')
        return

    years = glob.glob(args['reprocess_dir'] + "/*/")

    for year in years:
        experiments = glob.glob(year + "/*/")
        
        for experiment in experiments:
            args['data'] = experiment + "pickle/*.pickle"
            args['outdir'] = experiment + "reports/" + args['reprocess_version']

            logging.info("Processing {experiment}".format(experiment=experiment))
            analyse_all_data(args)


def analyse_all_data(args):
    """
    Scan directory for pickle experiment files
    :param args:
    :return:
    """
    start_time = time.time()  # record time for performance evaluation
    files_to_process = []
    for py in glob.iglob(args['data'], recursive=True):
        files_to_process.append(Path(py))

    if files_to_process == []:
        logging.error('Error: No files to process in directory ', args['data'])
        return

    logging.info('Found ' + str(len(files_to_process)) + ' files to process')

    base_outdir = args['outdir']

    mp_batches = []
    for file in files_to_process:
        label = os.path.basename(file).split('.')[0]
        basedir = os.path.dirname(file)

        if base_outdir is None:
            base_outdir = basedir
            logging.warning("--outdir not set, setting to --data folder")

        args['outdir'] = create_result_dirs(args, base_outdir, label)
        args['label'] = label
        args['basedir'] = base_outdir
        args['run_cmd'] = " ".join(sys.argv)
        args['file'] = file
        mp_batches.append(dict(args))

        logging.info('Processing File: {file}'.format(file=file))

    with multiprocessing.Pool(args['cores']) as pool:
        results = list(pool.imap(analyse_experiment, mp_batches))

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info('Run took ' + str(round(processing_time, 1)) + ' sec')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Path to directories
    parser.add_argument('--data',              default=None,
                                               help='Directory containing files to be processed -- Passed as globs')

    parser.add_argument('--outdir',            default=None,
                                               help='Directory to store result files in.')

    parser.add_argument('--reprocess_dir',     default='/data/MLIA_active_data/data_OWLS/ACME/lab_data/',
                                               help='Top level lab data directory for bulk reprocessing')

    parser.add_argument('--reprocess_version', default=None,
                                               help='Version tag for bulk reprocessing.')

    # Internal files
    parser.add_argument('--masses',            default='configs/compounds.yml',
                                               help='Path to file containing known masses')

    parser.add_argument('--params',            default='configs/acme_config.yml',
                                               help='Path to config file for Analyser')

    parser.add_argument('--sue_weights',       default='configs/acme_sue_weights.yml',
                                               help='Path to weights for Science Utility Estimate')

    parser.add_argument('--dd_weights',         default='configs/acme_dd_weights.yml',
                                                help='Path to weights for Diversity Descriptor')

    # Flags
    parser.add_argument('--noplots',           action='store_true',
                                               help='If True, does not save resulting plots')

    parser.add_argument('--noexcel',           action='store_true',
                                               help='If True, does not save final excel file containing analysis information')

    parser.add_argument('--debug_plots',       action='store_true',
                                               help='If True, generates extra plots for debugging purposes')

    parser.add_argument('--reprocess',         action='store_true',
                                               help='Bulk processing of the lab data store inside data_OWLS')

    parser.add_argument('--field_mode',        action='store_true',
                                                help='Only outputs science products')

    parser.add_argument('--cores',             type=int,
                                               help="How many processor cores to utilize",
                                               default=7)

    # Running options
    parser.add_argument('--knowntraces',       action='store_true',
                                               help='Process only known masses specified in configs/compounds.yml')

    args = parser.parse_args()

    # setup the plotter
    pltt = Plotter(save_to=op.join("output", "ACME_pipeline_memory.mp4"))
    globalQ = pltt.get_queues('ACME_pipeline.py')

    # Set up the watcher arguments
    watch = {'ACME_pipeline.py': {'queue': globalQ.graph, 'pid': os.getpid()}}

    # Start watcher then the plotter
    watcher(watch)
    pltt.start()

    if args.reprocess:
        bulk_reprocess(vars(args))
    else:
        analyse_all_data(vars(args))

    try:
        pltt.stop()
    except:
        logger.warning("Memory tracker failed to shut down correctly.")

    logging.info("======= Done =======")
