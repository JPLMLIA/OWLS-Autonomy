'''
Command line interface to the ACME data processing pipeline
'''
import sys
import logging
import argparse
import glob
import os
import os.path as op
import time
import multiprocessing

from pathlib import Path
from sys     import platform
from tqdm    import tqdm

from utils                           import logger
from utils.memory_tracker.plotter    import Plotter, watcher
from acme_cems.lib.analyzer          import analyse_experiment


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

def check_existing(args, basedir, label):
    '''Checks if this particular file already has output products'''

    # get appropriate output dir
    if (args['knowntraces']):
        outdir = op.join(basedir, label, 'Known_Masses')
    else:
        outdir = op.join(basedir, label, 'Unknown_Masses')

    # check if dir has a manifest (last step in pipeline)
    return op.isfile(op.join(outdir, label+'_manifest.json'))

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

            logging.info(f"Processing {experiment}")
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
        logging.error(f"Error: No files to process in directory {args['data']}")
        return

    logging.info('Found ' + str(len(files_to_process)) + ' files to process')

    base_outdir = args['outdir']

    mp_batches = []
    for f in files_to_process:
        label = Path(f).stem
        basedir = os.path.dirname(f)

        if base_outdir is None:
            base_outdir = basedir
            logging.warning("--outdir not set, setting to --data folder")
        
        if not check_existing(args, base_outdir, label) or not args['skip_existing']:
            args['outdir'] = create_result_dirs(args, base_outdir, label)
            args['label'] = label
            args['basedir'] = base_outdir
            args['run_cmd'] = " ".join(sys.argv)
            args['file'] = f
            mp_batches.append(dict(args))

            logging.info(f'Processing File: {f}')
        else:
            logging.warning(f'Skipping Existing File: {f}')

    if len(mp_batches) != 0:
        #with multiprocessing.Pool(args['cores']) as pool:
        #    _ = list(pool.imap(analyse_experiment, mp_batches))
        for b in mp_batches:
            analyse_experiment(b)

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info('Run took ' + str(round(processing_time, 1)) + ' sec')


def main():

    parser = argparse.ArgumentParser()

    # Path to directories
    parser.add_argument('--data',              default=None,
                                               help='Glob of files to be processed')

    parser.add_argument('--outdir',            default=None,
                                               help='Directory to store result files in.')

    parser.add_argument('--reprocess_dir',     default='/data/MLIA_active_data/data_OWLS/ACME/lab_data/',
                                               help='Top level lab data directory for bulk reprocessing')

    parser.add_argument('--reprocess_version', default=None,
                                               help='Version tag for bulk reprocessing.')

    # Internal files
    parser.add_argument('--masses',             default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'compounds.yml'),
                                                help='Path to file containing known masses. Default is cli/configs/compounds.yml')

    parser.add_argument('--params',             default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'acme_config.yml'),
                                                help='Path to config file for Analyser. Default is cli/configs/acme_config.yml')

    parser.add_argument('--sue_weights',        default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'acme_sue_weights.yml'),
                                                help='Path to weights for Science Utility Estimate. Default is cli/configs/acme_sue_weights.yml')

    parser.add_argument('--dd_weights',         default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'acme_dd_weights.yml'),
                                                help='Path to weights for Diversity Descriptor. Default is cli/configs/acme_dd_weights.yml')

    parser.add_argument('--log_name',           default="ACME_pipeline.log",
                                                help="Filename for the pipeline log. Default is ACME_pipeline.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    # Flags
    parser.add_argument('--noplots',            action='store_true',
                                                help='If True, does not save resulting plots')

    parser.add_argument('--noexcel',            action='store_true',
                                                help='If True, does not save final excel file containing analysis information')

    parser.add_argument('--debug_plots',        action='store_true',
                                                help='If True, generates extra plots for debugging purposes')

    parser.add_argument('--reprocess',          action='store_true',
                                                help='Bulk processing of the lab data store inside data_OWLS')
    
    parser.add_argument('--skip_existing',      action='store_true',
                                                help='Skip existing reports when bulk reprocessing')

    parser.add_argument('--field_mode',         action='store_true',
                                                help='Only outputs science products')

    parser.add_argument('--cores',              type=int,
                                                help="How many processor cores to utilize",
                                                default=7)

    parser.add_argument('--saveheatmapdata',    action='store_true',
                                                help="Saves the heatmap as a data file")

    parser.add_argument('--priority_bin',       default=0, type=int,
                                                help='Downlink priority bin in which to place generated products')

    parser.add_argument('--manifest_metadata',  default=None, type=str,
                                                help='Manifest metadata (YAML string); takes precedence over file entries')

    parser.add_argument('--manifest_metadata_file',
                                                default=None, type=str,
                                                help='Manifest metadata file (YAML)')

    # Running options
    parser.add_argument('--knowntraces',        action='store_true',
                                                help='Process only known masses specified in configs/compounds.yml')

    args = parser.parse_args()

    logger.setup_logger(args.log_name, args.log_folder)

    # setup the plotter
    pltt = Plotter(save_to=op.join(args.log_folder, "ACME_pipeline_memory.mp4"))
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
        ram_mean, ram_max = pltt.stop()
        logging.info(f'Average RAM:{ram_mean:.2f}GB, Max RAM:{ram_max:.2f}GB')
    except:
        logging.warning("Memory tracker failed to shut down correctly.")

    logging.info("======= Done =======")

if __name__ == "__main__":
    main()
