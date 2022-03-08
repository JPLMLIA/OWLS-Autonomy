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
from fsw.ACME.lib.analyzer           import analyse_experiment


def make_dirs(outdir):
    '''Create directory in path outdir '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)


def create_result_dirs(basedir, label, space_mode, knowntraces, debug_plots, saveheatmapdata, noplots):
    '''Creates directories where to save results depending on running options'''

    if (space_mode):
        outdir = os.path.join(basedir, label)
        make_dirs(os.path.join(outdir, "Mugshots"))
    else:
        if (knowntraces):
            outdir = os.path.join(basedir, label, 'Known_Masses')
        else:
            outdir = os.path.join(basedir, label, 'Unknown_Masses')

        make_dirs(outdir)
        make_dirs(os.path.join(outdir, "Heat_Maps"))
        make_dirs(os.path.join(outdir, "Mugshots"))
        if (debug_plots):
            make_dirs(os.path.join(outdir, "Debug_Plots"))

        if (saveheatmapdata) and not (knowntraces):
            make_dirs(os.path.join(outdir, "Data_Files"))

        if not (noplots):
            make_dirs(os.path.join(outdir, "Time_Trace"))
            make_dirs(os.path.join(outdir, "Mass_Spectra"))
        
    return outdir

def check_existing(basedir, label, knowntraces):
    '''Checks if this particular file already has output products'''

    # get appropriate output dir
    if (knowntraces):
        outdir = op.join(basedir, label, 'Known_Masses')
    else:
        outdir = op.join(basedir, label, 'Unknown_Masses')

    # check if dir has a manifest (last step in pipeline)
    return op.isfile(op.join(outdir, label+'_manifest.json'))


def ACME_ground(data, outdir, masses, params, sue_weights, dd_weights, log_name, log_folder, cores, priority_bin, manifest_metadata, manifest_metadata_file, knowntraces, noplots, noexcel, debug_plots, space_mode, saveheatmapdata):
    """
    Scan directory for pickle experiment files
    :param args:
    :return:
    """
    start_time = time.time()  # record time for performance evaluation

    logger.setup_logger(log_name, log_folder)

    files_to_process = []
    for py in glob.iglob(data, recursive=True):
        files_to_process.append(Path(py))

    if files_to_process == []:
        logging.error(f"Error: No files to process in directory {data}")
        return

    logging.info('Found ' + str(len(files_to_process)) + ' files to process')

    base_outdir = outdir

    mp_batches = []
    for f in files_to_process:
        label = Path(f).stem
        basedir = os.path.dirname(f)

        if base_outdir is None:
            base_outdir = basedir
            logging.warning("--outdir not set, setting to --data folder")
        
        outdir = create_result_dirs(base_outdir, label, space_mode, knowntraces, debug_plots, saveheatmapdata, noplots)
        label = label
        basedir = base_outdir
        run_cmd = " ".join(sys.argv)
        file = f
        mp_batches.append({'data':data, 
                           'outdir':outdir,
                           'masses':masses,
                           'params':params,
                           'sue_weights':sue_weights,
                           'dd_weights':dd_weights,
                           'log_name':log_name,
                           'log_folder':log_folder,
                           'noplots':noplots,
                           'noexcel':noexcel,
                           'debug_plots':debug_plots,
                           'space_mode':space_mode, 
                           'cores':cores, 
                           'saveheatmapdata':saveheatmapdata, 
                           'priority_bin':priority_bin, 
                           'manifest_metadata':manifest_metadata, 
                           'manifest_metadata_file':manifest_metadata_file, 
                           'knowntraces':knowntraces,
                           'file':file,
                           'label':label,
                           'basedir':basedir,
                           'run_cmd':run_cmd})

        logging.info(f'Processing File: {f}')

    if len(mp_batches) != 0:
        for b in mp_batches:
            analyse_experiment(b)

    end_time = time.time()
    processing_time = end_time - start_time
    logging.info('Run took ' + str(round(processing_time, 1)) + ' sec')

    # Shut down all open loggers so they do not interfere with future runs in the same session
    for x in range(0, len(logging.getLogger().handlers)):
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def main():

    parser = argparse.ArgumentParser()

    # Path to directories
    parser.add_argument('--data',              default=None,
                                               help='Glob of files to be processed')

    parser.add_argument('--outdir',            default=None,
                                               help='Directory to store result files in.')

    # Internal files
    parser.add_argument('--masses',             default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'compounds.yml'),
                                                help='Path to file containing known masses. Default is cli/configs/compounds.yml')

    parser.add_argument('--params',             default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'acme_config.yml'),
                                                help='Path to config file for Analyser. Default is cli/configs/acme_config.yml')

    parser.add_argument('--sue_weights',        default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'acme_sue_weights.yml'),
                                                help='Path to weights for Science Utility Estimate. Default is cli/configs/acme_sue_weights.yml')

    parser.add_argument('--dd_weights',         default=op.join(op.abspath(op.dirname(__file__)), 'configs', 'acme_dd_weights.yml'),
                                                help='Path to weights for Diversity Descriptor. Default is cli/configs/acme_dd_weights.yml')

    parser.add_argument('--log_name',           default="ACME_ground_pipeline.log",
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

    parser.add_argument('--space_mode',         action='store_true',
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

    # setup the plotter
    pltt = Plotter(save_to=op.join(args.log_folder, "ACME_ground_pipeline_memory.mp4"))
    globalQ = pltt.get_queues('ACME_ground_pipeline.py')

    # Set up the watcher arguments
    watch = {'ACME_ground_pipeline.py': {'queue': globalQ.graph, 'pid': os.getpid()}}

    # Start watcher then the plotter
    watcher(watch)
    pltt.start()

    ACME_ground(args.data, 
                args.outdir, 
                args.masses, 
                args.params, 
                args.sue_weights, 
                args.dd_weights, 
                args.log_name, 
                args.log_folder, 
                args.cores, 
                args.priority_bin, 
                args.manifest_metadata, 
                args.manifest_metadata_file, 
                args.knowntraces, 
                args.noplots, 
                args.noexcel, 
                args.debug_plots, 
                args.space_mode, 
                args.saveheatmapdata)

    try:
        ram_mean, ram_max = pltt.stop()
        logging.info(f'Average RAM:{ram_mean:.2f}GB, Max RAM:{ram_max:.2f}GB')
    except:
        logging.warning("Memory tracker failed to shut down correctly.")

    logging.info("======= Done =======")

if __name__ == "__main__":
    main()
