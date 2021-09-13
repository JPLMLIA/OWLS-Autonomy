'''
Command line interface to the HiRAILS data processing pipeline
'''
import sys
import logging
import argparse
import timeit
import os
import os.path as op
import yaml
import glob
from datetime import datetime
import errno
from pathlib import Path
import string
import copy
from cli.HELM_pipeline import MANIFEST_STEP

from utils                         import logger
from utils import manifest
from utils.manifest                import AsdpManifest, load_manifest_metadata
from utils.memory_tracker.plotter  import Plotter, watcher

from helm_dhm.validate import process
from helm_dhm.validate import utils
from utils.dir_helper  import get_batch_subdir, get_exp_subdir

from hirails_hrfi.tracker.tracker               import hrfi_tracker
from hirails_hrfi.asdp.asdp                     import mugshots, generate_SUEs_DDs

TRACKER_STEP = "tracker"
ASDP_STEP = "asdp"
MANIFEST_STEP = "manifest"
PIPELINE = "pipeline"

### Pipeline Steps ###

def tracker_experiment(experiment, config):
    '''Run the tracker on experiment'''
    holograms = glob.glob(op.join(experiment, config['experiment_dirs']['hologram_dir'], "*"))
    for hologram in holograms:
        hrfi_tracker(hologram, experiment, config)

def asdp_experiment(experiment, config):
    '''Create asdp's for experiment'''
    asdp_dir = get_exp_subdir('asdp_dir', experiment, config, rm_existing=True)
    holograms = process.get_files(experiment, config)

    for hologram in holograms:
        mugshots(hologram, experiment, config)

    mugshot_ids = sorted([Path(x).stem for x in glob.glob(op.join(asdp_dir, 'mugshots', '*.tif'))])
    for mid in mugshot_ids:
        generate_SUEs_DDs(asdp_dir, mid, config['sue'], config['dd'])

    num_files = len(holograms)
    return num_files

def asdp_batch(inputs, _, batch_outdir, config):
    '''Create batch asdp's from singular experiment products'''
    total_files = 0
    frame_rate = 15 # TODO - Put in config? Argparse? Read from experiment metadata?
    total_files = sum(inputs)
    capture_time = total_files / frame_rate
    run_time = timeit.default_timer() - start_time
    performance_ratio = run_time / capture_time
    logging.info("Runtime Performance Ratio: {ratio:.1f}x (Data Processing Time / Raw Data Creation Time)".format(ratio=performance_ratio))

def manifest_experiment(experiment, config):
    '''Create manifest for experiment'''
    asdp_dir = get_exp_subdir('asdp_dir', experiment, config, rm_existing=False)
    track_dir = get_exp_subdir('track_dir', experiment, config, rm_existing=False)

    priority_bin = config.get('_priority_bin', 0)
    metadata = config.get('_manifest_metadata', {})

    output_dir = op.join(asdp_dir, 'manifests')
    if not op.exists(output_dir):
        os.mkdir(output_dir)

    mugshot_ids = sorted([Path(x).stem for x in glob.glob(op.join(asdp_dir, 'mugshots', '*.tif'))])
    for mid in mugshot_ids:
        manifest = AsdpManifest('hirails', priority_bin)
        manifest.add_metadata(**metadata)

        # Track products
        manifest.add_entry(
            'bounding_boxes',
            'tracks',
            op.join(track_dir, mid.split('_')[0]+'_bboxes.json')
        )

        # ASDPs
        manifest.add_entry(
            'mugshot',
            'asdp',
            op.join(asdp_dir, 'mugshots', mid+'.tif')
        )
        manifest.add_entry(
            'binary_mugshot',
            'asdp',
            op.join(asdp_dir, 'binary_mugshots', mid+'_binary.tif')
        )
        manifest.add_entry(
            'ellipse',
            'asdp',
            op.join(asdp_dir, 'ellipses', mid+'_hull.json')
        )
        manifest.add_entry(
            'contours',
            'asdp',
            op.join(asdp_dir, 'contours', mid+'_contour.json')
        )
        manifest.add_entry(
            'diversity_descriptor',
            'metadata',
            op.join(asdp_dir, 'DDs', mid+'_dd.csv')
        )
        manifest.add_entry(
            'science_utility',
            'metadata',
            op.join(asdp_dir, 'SUEs', mid+'_sue.csv')
        )

        manifest.write(op.join(output_dir, mid+'_manifest.json'))

### Pipeline Helpers ###

def get_override_config(default_config, new_config):
    '''Returns new config wtih all key-values in default_config overrident by those matching in new_config'''
    config = copy.deepcopy(default_config)
    _override_config(config, new_config)
    return config

def _override_config(default_config, new_config, prefix=None):
    '''Recursively overrides all key-values in default_config matching those in new_config'''
    if prefix is None:
        prefix = []

    for key in new_config.keys():
        if isinstance(new_config[key], dict):
            p = prefix[:]
            p.append(key)
            _override_config(default_config, new_config[key], p)
        else:
            subdict = default_config
            has_prefix = True
            for k in prefix:
                if k not in subdict.keys() or not isinstance(subdict[k], dict):
                    has_prefix = False
                    subdict[k] = {}
                subdict = subdict[k]
            if not has_prefix or key not in subdict:
                logging.warning("Creating new config key: {}".format(prefix + [key]))
            subdict[key] = new_config[key]

def has_experiment_outputs(step, experiment, config):
    '''Returns true if experiment already has all expected step outputs'''

    # The expected output directories for each step; a step is rerun if any are empty
    experiment_directories = {TRACKER_STEP : ['track_dir', 'evaluation_dir'],
                              ASDP_STEP : ['asdp_dir']}

    for directory in experiment_directories[step]:
        exp_dir = get_exp_subdir(directory, experiment, config)
        if not op.isdir(exp_dir) or len(os.listdir(exp_dir)) == 0:
            logging.warning("\tStep {} does not have output {} at {}!".format(step, directory, exp_dir))
            return False

    return True

def get_timestamp(step, experiment, config):
    '''Return the timestamp of last run of step on experiment'''
    timestamp_dir = get_exp_subdir('timestamp_dir', experiment, config)
    try:
        with open(op.join(timestamp_dir, step), 'r') as ts_file:
            step_ts = int(ts_file.readline())
        return step_ts
    except:
        logging.warning("No timestamp found in experiment {} for step {}".format(
            experiment, step
        ))

def should_run(step, use_preexisting, experiment, config):
    '''Determine if step needs to be rerun on experiment'''

    # Run if caching disabled
    if not use_preexisting:
        return True

    # Run if outputs don't exists, or were run with a different config
    if not has_experiment_outputs(step, experiment, config):
        return True

    # TODO: Run if config doesn't match previous run

    # Mapping from each step to the exhaustive list of steps that should trigger a rerun
    experiment_dependencies = {TRACKER_STEP : [],
                               ASDP_STEP : [TRACKER_STEP]}

    # Rerun if any of of the steps depended on by this step were run more recently
    step_ts = get_timestamp(step, experiment, config)
    if not step_ts:
        return True
    for dependency in experiment_dependencies[step]:
        dep_ts = get_timestamp(dependency, experiment, config)
        if not dep_ts or dep_ts > step_ts:
            return True

    return False

def parse_steps(cli_args):
    '''Parses command line steps/pipeline keywords and returns list of steps to run.
       Step tuples include name of step, functions associated with step, and whether step can use existing products'''
    step_names = cli_args.steps
    use_existing = cli_args.use_existing

    cache_allowed_steps = [TRACKER_STEP, ASDP_STEP]

    step_mappings = {
        TRACKER_STEP :      [tracker_experiment, None, None],
        ASDP_STEP :         [asdp_experiment, asdp_batch, None],
        MANIFEST_STEP :     [manifest_experiment, None, None]
    }

    pipelines = {
        PIPELINE :        [TRACKER_STEP, ASDP_STEP, MANIFEST_STEP]
    }

    # Convert pipelines to steps
    if len(step_names) == 1 and step_names[0] in pipelines:
        step_names = pipelines[step_names[0]]

    # Create step tuples of the form: (step_name, exp_func, batch_func, get_previous_func, can_reuse)
    step_tuples = []
    for step_name in step_names:
        if step_name not in step_mappings:
            raise Exception("Unrecognized step or pipeline keyword: {}".format(step_name))
        step = [step_name]
        step.extend(step_mappings[step_name])
        step.append(step_name in use_existing and step_name in cache_allowed_steps)
        step_tuples.append(tuple(step))

    can_reuse = []
    for st in step_tuples:
        if st[-1]:
            can_reuse.append(st[0])
    if len(can_reuse) > 1:
        logging.info("USE EXISTING ENABLED FOR STEPS: {}".format(' '.join(can_reuse)))

    return step_tuples

def pipeline_run_step(step_tuple, experiments, batch_outdir, config):
    """Executes a step in the pipeline, managing generic logging and reuse of intermediate results

    Parameters
    ----------
    step_tuple : tuple
        step_name : str
            The name of the step to execute
        experiment_func : function
            Function to run on each experiment. Should take experiment string as argument and return
        batch_func : function
            Function to run on the results of all experiments for this step
        get_preexisting_func : function
            Function to get a preexisting result (if needed for batch) when experiment_func is skipped on an experiment
        use_existing : bool
            Whether this step can skipped using preexisting results
    experiments : list
        The experiments used in this run
    batch_outdir : str
        The path to the batch output directory
    config : dict
        The config for this run
    """

    # Parse step tuple
    step = step_tuple[0]
    experiment_func = step_tuple[1]
    batch_func = step_tuple[2]
    get_preexisting_func = step_tuple[3]
    use_preexisting =  step_tuple[4]

    logging.info("Beginning {} step...".format(step))
    st = timeit.default_timer()
    outputs = []
    # Run per experiment steps (if any)
    if experiment_func:
        for experiment in experiments:
            # Skip running on an experiment if we can use a pre-existing result
            if not should_run(step, use_preexisting, experiment, config):
                logging.info("Using cached {} result for experiment {}".format(step, experiment))
                if get_preexisting_func:
                    outputs.append(get_preexisting_func(experiment, config))
                continue
            # Else run and timestamp
            result = experiment_func(experiment, config)
            ct = int(datetime.utcnow().timestamp())
            timestamp_dir = get_exp_subdir('timestamp_dir', experiment, config)
            with open(op.join(timestamp_dir, step), 'w') as ts_file:
                ts_file.write(str(ct))
            #TODO: Dump subset of config depended on by this step
            if result:
                outputs.append(result)

    # Run any batch operations on the outputs of all experiments
    if batch_func:
        batch_func(outputs, experiments, batch_outdir, config)

    logging.info("Finished {} step. Elapsed time = {time:.2f} s".format(
        step, time=timeit.default_timer() - st))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',             default=op.join(op.abspath(op.dirname(__file__)), "configs", "hirails_config.yml"),
                                                help="Path to configuration file. Default is cli/configs/hiralis_config.yml")

    parser.add_argument('--experiments',        nargs='+',
                                                required=True,
                                                help="Glob-able string patterns indicating sets of data files to be processed.")

    all_steps = [TRACKER_STEP, ASDP_STEP, MANIFEST_STEP]
    pipeline_keywords = [PIPELINE]
    steps_options = all_steps + pipeline_keywords
    cache_allowed_steps = [TRACKER_STEP, ASDP_STEP]

    parser.add_argument('--use_existing',       default=[], nargs='+',
                                                required=False,
                                                choices=cache_allowed_steps,
                                                help="Allow reusing intermediate experiment results from previous runs for these steps.")

    parser.add_argument('--steps',              nargs='+',
                                                required=True,
                                                choices=steps_options,
                                                help=" | ".join(steps_options))

    parser.add_argument('--cores',              type=int,
                                                help="How many processor cores to utilize",
                                                default=7)

    parser.add_argument('--batch_outdir',       required=True,
                                                help="Directory to write batch results")

    parser.add_argument('--note',               default="",
                                                help="Note to be appended to batch outdir name")

    parser.add_argument('--log_name',           default="HIRAILS_pipeline.log",
                                                help="Filename for the pipeline log. Default is HIRAILS_pipeline.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    parser.add_argument('--field_mode',         action='store_true',
                                                help='Only outputs field products')

    parser.add_argument('--priority_bin',       default=0, type=int,
                                                help='Downlink priority bin in which to place generated products')

    parser.add_argument('--manifest_metadata',  default=None, type=str,
                                                help='Manifest metadata (YAML string); takes precedence over file entries')

    parser.add_argument('--manifest_metadata_file',
                                                default=None, type=str,
                                                help='Manifest metadata file (YAML)')
    args = parser.parse_args()

    logger.setup_logger(args.log_name, args.log_folder)

    steps_to_run = parse_steps(args)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logging.info("Loaded config.")

    manifest_metadata = load_manifest_metadata(
        args.manifest_metadata_file, args.manifest_metadata)
    
    # To keep pipeline step calling convention simpler, add one-off args to config here
    config['_cores'] = args.cores
    config['_field_mode'] = args.field_mode
    config['_priority_bin'] = args.priority_bin
    config['_manifest_metadata'] = manifest_metadata

    # setup batch outdir parent directory
    try:
        # try-catching avoids race condition
        os.mkdir(args.batch_outdir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        else:
            logging.info("Using existing batch output parent directory.")
            pass

    # setup batch outdir directory
    if config['raw_batch_dir']:
        # Absolute path for TOGA
        if args.note != "":
            logging.warning("Using raw batch dir, ignoring --note")
        batch_outdir = args.batch_outdir
    else:
        # Timestamped path for standard use
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.note != "":
            batch_outdir = op.join(args.batch_outdir, timestamp+"_"+args.note)
        else:
            batch_outdir = op.join(args.batch_outdir, timestamp)
        utils._check_create_delete_dir(batch_outdir, overwrite=False)

    # setup the plotter
    pltt = Plotter(save_to=op.join(args.log_folder, "HIRAILS_pipeline_memory.mp4"))
    globalQ = pltt.get_queues('HIRAILS_pipeline.py')

    # Set up the watcher arguments
    watch = {'HIRAILS_pipeline.py': {'queue': globalQ.graph, 'pid': os.getpid()}}

    # Start watcher then the plotter
    watcher(watch)
    pltt.start()

    global start_time
    start_time = timeit.default_timer()

    experiments = process.get_experiments(args.experiments, config)

    logging.info("Retrieved experiment dirs.")

    if experiments:
        # Run the pipeline
        for step_tuple in steps_to_run:
            pipeline_run_step(step_tuple, experiments, batch_outdir, config)
    else:
        logging.warning("No experiments found!")

    run_time = timeit.default_timer() - start_time

    try:
        ram_mean, ram_max = pltt.stop()
        logging.info(f'Average RAM:{ram_mean:.2f}GB, Max RAM:{ram_max:.2f}GB')
    except:
        logging.warning("Memory tracker failed to shut down correctly.")

    logging.info("Full script run time: {time:.1f} seconds".format(time=run_time))
    logging.info("======= Done =======")

if __name__ == "__main__":
    main()
