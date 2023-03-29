'''
Command line interface to the FAME data processing pipeline
'''
import matplotlib
matplotlib.use('Agg') # Agg backend required for F' integration

import sys
import logging
import argparse
import timeit
import os
import os.path as op
import yaml
import glob
from datetime import datetime
import json
import errno
from pathlib import Path
import string
import copy
import shutil

from utils                          import logger
from utils.manifest                 import AsdpManifest, load_manifest_metadata
from utils.memory_tracker.plotter   import Plotter, watcher

from fsw.HELM_FAME                  import validate
from fsw.HELM_FAME                  import utils
from fsw.HELM_FAME                  import preproc
from utils.dir_helper               import get_batch_subdir, get_exp_subdir

from fsw.HELM_FAME.LAP_tracker      import run_tracker
from fsw.HELM_FAME.tracker          import run_tracker as run_proj_tracker
from fsw.HELM_FAME.features         import get_features
from fsw.HELM_FAME.classifier       import predict, predict_batch_metrics
from fsw.HELM_FAME.asdp             import mugshots, generate_SUEs, generate_DDs
from utils.pipelines                import get_override_config, pipeline_run_step

PREPROC_STEP = "preproc"
VALIDATE_STEP = "validate"
TRACKER_STEP = "tracker"
PROJ_TRACKER_STEP = "proj_tracker"
FEATURES_STEP = "features"
PREDICT_STEP = "predict"
ASDP_STEP = "asdp"
MANIFEST_STEP = "manifest"
PIPELINE_PREDICT = "pipeline_predict"
PIPELINE_PRODUCTS = "pipeline_products"
PIPELINE_SPACE = "pipeline_space"

### Pipeline Steps ###

def preproc_experiment(experiment, config):
    '''Preprocess hologram files'''
    files = validate.get_files(experiment, config)
    preproc.resize_holograms(holo_fpaths=files,
                             outdir=get_exp_subdir('preproc_dir', experiment, config, rm_existing=True),
                             raw_shape=config['raw_hologram_resolution'],
                             resize_shape=config['preproc_resolution'],
                             n_workers=config['_cores'])

def validate_experiment(experiment, config):
    '''Create per experiment validate products'''
    files = validate.get_files(experiment, config)
    preproc_files = validate.get_preprocs(experiment, config)
    validate.validate_data_flight(exp_dir=experiment,
                                 holo_fpaths=files,
                                 preproc_fpaths=preproc_files,
                                 n_workers=config['_cores'],
                                 config=config,
                                 instrument="FAME")

def validate_batch(_, experiments, batch_outdir, config):
    '''Calculate global statistics'''
    validate.global_stats(exp_dirs=experiments,
                            out_dir=batch_outdir,
                            config=config)

def proj_tracker_experiment(experiment, config):
    '''Run the tracker on experiment'''
    files = validate.get_files(experiment, config)
    preproc_files = validate.get_preprocs(experiment, config)
    run_proj_tracker(exp_dir=experiment,
                holograms=preproc_files,
                originals=files,
                config=config,
                n_workers=config['_cores'])

def tracker_experiment(experiment, config):
    '''Run the tracker on experiment'''
    files = validate.get_files(experiment, config)
    preproc_files = validate.get_preprocs(experiment, config)
    run_tracker(exp_dir=experiment,
                holograms=preproc_files,
                originals=files,
                config=config,
                n_workers=config['_cores'])

def features_experiment(experiment, config):
    '''Compute features on experiment'''
    data_track_features = get_features(experiment=experiment,
                                       config=config,
                                       save=True,
                                       labeled=False)
    if not data_track_features:
        logging.error(f'Could not extract features for experiment {experiment}')
        return

def predict_experiment(experiment, config):
    '''Run predict on experiment'''
    return predict(experiment, config)

def predict_batch(inputs, experiments, batch_outdir, config):
    # Unmarshal experiment results
    batch_true_Y = []
    batch_pred_Y = []
    batch_prob_Y = []
    batch_alltracks = 0
    for input in inputs:
        batch_true_Y.extend(input[0])
        batch_pred_Y.extend(input[1])
        batch_prob_Y.extend(input[2])
        batch_alltracks += input[3]
    predict_batch_metrics(batch_true_Y, batch_pred_Y, batch_prob_Y, batch_alltracks,
                          batch_outdir, config)

def asdp_experiment(experiment, config):
    '''Create asdp's for experiment'''
    asdp_dir = get_exp_subdir('asdp_dir', experiment, config, rm_existing=True)
    validate_dir = get_exp_subdir('validate_dir', experiment, config)
    predict_dir = get_exp_subdir('predict_dir', experiment, config)
    feature_dir = get_exp_subdir('features_dir', experiment, config)
    feat_file = op.join(feature_dir, config['features']['output'])
    track_fpaths = sorted(glob.glob(op.join(predict_dir, '*.json')))
    holograms = validate.get_files(experiment, config)
    num_files = len(holograms)

    # Make a copy of DQE file in newly created asdp folder for sync to main PC
    dqe_validate_path = op.join(validate_dir, experiment.split("/")[-1]) + "_dqe.csv"
    dqe_asdp_path = op.join(asdp_dir, experiment.split("/")[-1]) + "_dqe.csv"
    if os.path.exists(dqe_validate_path):
        shutil.copy(dqe_validate_path, dqe_asdp_path)

    mugshots(experiment, holograms, experiment, os.path.join(asdp_dir,"mugshots"), config)
    generate_SUEs(experiment, asdp_dir, track_fpaths, config)
    generate_DDs(experiment, asdp_dir, track_fpaths, config['dd'], feat_file)

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
    exp_name = Path(experiment).name

    validate_dir = get_exp_subdir('validate_dir', experiment, config)
    predict_dir = get_exp_subdir('predict_dir', experiment, config)
    asdp_dir = get_exp_subdir('asdp_dir', experiment, config)

    priority_bin = config.get('_priority_bin', 0)
    metadata = config.get('_manifest_metadata', {})

    manifest = AsdpManifest('fame', priority_bin)
    manifest.add_metadata(**metadata)

    # validate products
    manifest.add_entry(
        'processing_report',
        'validate',
        op.join(validate_dir, exp_name + '_processing_report.txt'),
    )
    manifest.add_entry(
        'timestats_density',
        'validate',
        op.join(validate_dir, exp_name + '_timestats_density.csv'),
    )
    manifest.add_entry(
        'timestats_max_intensity',
        'validate',
        op.join(validate_dir, exp_name + '_timestats_max_intensity.csv'),
    )
    manifest.add_entry(
        'timestats_pixeldiff',
        'validate',
        op.join(validate_dir, exp_name + '_timestats_pixeldiff.csv'),
    )
    manifest.add_entry(
        'mhi_image_info',
        'validate',
        op.join(validate_dir, exp_name + '_mhi.jpg'),
    )

    # predicted path products
    # note that we're listing predict step output, not tracker output.
    manifest.add_entry(
        'predicted_tracks',
        'predict',
        op.join(predict_dir),
    )

    # asdp products
    manifest.add_entry(
        'track_mugshots',
        'asdp',
        op.join(asdp_dir, 'mugshots'),
    )
    manifest.add_entry(
        'diversity_descriptor',
        'metadata',
        op.join(asdp_dir, exp_name + '_dd.csv'),
    )
    manifest.add_entry(
        'science_utility',
        'metadata',
        op.join(asdp_dir, exp_name + '_sue.csv'),
    )
    manifest.add_entry(
        'data_quality',
        'metadata',
        op.join(asdp_dir, exp_name + '_dqe.csv'),
    )

    manifest.write(op.join(asdp_dir, exp_name + '_manifest.json'))


### Pipeline Helpers ###

def parse_steps(step_names, use_existing, predict_model, space_mode):
    '''Parses command line steps/pipeline keywords and returns list of steps to run.
       Step tuples include name of step, functions associated with step, and whether step can use existing products'''

    cache_allowed_steps = [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP]

    step_mappings = {
        PREPROC_STEP :      [preproc_experiment, None, None],
        VALIDATE_STEP :     [validate_experiment, validate_batch, None],
        PROJ_TRACKER_STEP : [proj_tracker_experiment, None, None],
        TRACKER_STEP :      [tracker_experiment, None, None],
        FEATURES_STEP :     [features_experiment, None, None],
        PREDICT_STEP :      [predict_experiment, predict_batch, None],
        ASDP_STEP :         [asdp_experiment, asdp_batch, None],
        MANIFEST_STEP :     [manifest_experiment, None, None]
    }

    pipelines = {
        PIPELINE_PREDICT :      [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, FEATURES_STEP, PREDICT_STEP],
        PIPELINE_PRODUCTS :     [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, FEATURES_STEP, PREDICT_STEP,
                                 ASDP_STEP, MANIFEST_STEP],
        PIPELINE_SPACE :        [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, FEATURES_STEP,
                                 PREDICT_STEP, ASDP_STEP, MANIFEST_STEP]
    }

    # Pipeline-wise check
    if PIPELINE_SPACE in step_names and not space_mode:
        logging.error("--steps pipeline_space requires --space_mode")

    # Convert pipelines to steps
    if len(step_names) == 1 and step_names[0] in pipelines:
        step_names = pipelines[step_names[0]]

    # Various checks after substituting pipeline keywords
    if PREDICT_STEP in step_names and predict_model == "":
        logging.error("--steps predict requires --predict_model")

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

def FAME_flight(config, experiments, steps, use_existing, log_name, log_folder, cores, predict_model, space_mode, batch_outdir, priority_bin=0, manifest_metadata_file=None, manifest_metadata=None, note="", kill_file="FAME_flight_kill_file"):
    """Main logic function for the FAME flight software"""

    global start_time
    start_time = timeit.default_timer()

    logger.setup_logger(log_name, log_folder)

    steps_to_run = parse_steps(steps, use_existing, predict_model, space_mode)

    with open(config) as f:
        config = yaml.safe_load(f)

    logging.info("Loaded config.")

    manifest_metadata = load_manifest_metadata(manifest_metadata_file, manifest_metadata)

    # Mapping from each step to the exhaustive list of steps that should trigger a rerun
    exp_deps = {PREPROC_STEP : [],
                VALIDATE_STEP : [PREPROC_STEP],
                TRACKER_STEP : [VALIDATE_STEP, PREPROC_STEP],
                FEATURES_STEP : [TRACKER_STEP]}

    # The expected output directories for each step; a step is rerun if any are empty
    exp_dirs = {PREPROC_STEP : ['preproc_dir'],
                VALIDATE_STEP : ['validate_dir'],
                TRACKER_STEP : ['track_dir', 'evaluation_dir'],
                PROJ_TRACKER_STEP : ['track_dir', 'evaluation_dir'],
                PREDICT_STEP : ['predict_dir'],
                FEATURES_STEP : ['features_dir'],
                ASDP_STEP : ['asdp_dir']}

    exp_paths = {}

    # To keep pipeline step calling convention simpler, add one-off args to config here
    config['_cores'] = cores
    config['_model_absolute_path'] = predict_model
    config['_space_mode'] = space_mode
    config['_priority_bin'] = priority_bin
    config['_manifest_metadata'] = manifest_metadata
    config['_kill_file'] = kill_file

    # setup batch outdir parent directory
    try:
        # try-catching avoids race condition
        os.mkdir(batch_outdir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        else:
            logging.info("Using existing batch output parent directory.")
            pass

    # setup batch outdir directory
    if config['raw_batch_dir']:
        # Absolute path for TOGA
        if note != "":
            logging.warning("Using raw batch dir, ignoring --note")
    else:
        # Timestamped path for standard use
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if note != "":
            batch_outdir = op.join(batch_outdir, timestamp+"_"+note)
        else:
            batch_outdir = op.join(batch_outdir, timestamp)
        utils._check_create_delete_dir(batch_outdir, overwrite=False)

    experiments = validate.get_experiments(experiments, config)

    logging.info("Retrieved experiment dirs.")

    if experiments:
        # Run the pipeline
        for step_tuple in steps_to_run:
            pipeline_run_step(step_tuple, experiments, batch_outdir, exp_deps, exp_dirs, exp_paths, config)
    else:
        logging.error("No experiments found!")

    run_time = timeit.default_timer() - start_time

    logging.info("Full script run time: {time:.1f} seconds".format(time=run_time))

    # Shut down all open loggers so they do not interfere with future runs in the same session
    for x in range(0, len(logging.getLogger().handlers)):
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',             default=op.join(op.abspath(op.dirname(__file__)), "configs", "fame_config_rgb.yml"),
                                                help="Path to configuration file. Default is cli/configs/fame_config_rgb.yml")

    parser.add_argument('--experiments',        nargs='+',
                                                required=True,
                                                help="Glob-able string patterns indicating sets of data files to be processed.")

    all_steps = [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, PROJ_TRACKER_STEP, FEATURES_STEP, PREDICT_STEP, ASDP_STEP, MANIFEST_STEP]
    pipeline_keywords = [PIPELINE_PREDICT, PIPELINE_PRODUCTS, PIPELINE_SPACE]
    steps_options = all_steps + pipeline_keywords
    cache_allowed_steps = [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP]

    parser.add_argument('--use_existing',       default=[], nargs='+',
                                                required=False,
                                                choices=cache_allowed_steps,
                                                help=f"Steps for which to use existing output: [{', '.join(cache_allowed_steps)}]",
                                                metavar='CACHED_STEPS')

    parser.add_argument('--steps',              nargs='+',
                                                required=True,
                                                choices=steps_options,
                                                help=f"Steps to run in the pipeline: [{', '.join(steps_options)}]",
                                                metavar='STEPS')

    parser.add_argument('--cores',              type=int,
                                                help="How many processor cores to utilize",
                                                default=7)

    parser.add_argument('--batch_outdir',       required=True,
                                                help="Directory to write batch results")

    parser.add_argument('--note',               default="",
                                                help="Note to be appended to batch outdir name")

    parser.add_argument('--log_name',           default="FAME_flight_pipeline.log",
                                                help="Filename for the pipeline log. Default is FAME_flight_pipeline.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    parser.add_argument('--kill_file',          default="FAME_flight_kill_file",
                                                help="Pipeline checks for this file between steps and halts if found")

    parser.add_argument('--predict_model',      default=op.join(op.abspath(op.dirname(__file__)), "models", "classifier_labelbox_RF_v03.pickle"),
                                                help="Path to the pretrained model for prediction. Default is models/classifier_labelbox_RF_v03.pickle")

    parser.add_argument('--space_mode',         action='store_true',
                                                help='Only outputs space products')

    parser.add_argument('--priority_bin',       default=2, type=int,
                                                help='Downlink priority bin in which to place generated products. Defaults to 2')

    parser.add_argument('--manifest_metadata',  default=None, type=str,
                                                help='Manifest metadata (YAML string); takes precedence over file entries')

    parser.add_argument('--manifest_metadata_file',
                                                default=None, type=str,
                                                help='Manifest metadata file (YAML)')

    args = parser.parse_args()


    # setup the plotter
    pltt = Plotter(save_to=op.join(args.log_folder, "FAME_flight_pipeline_memory.mp4"))
    globalQ = pltt.get_queues('FAME_flight_pipeline.py')

    # Set up the watcher arguments
    watch = {'FAME_flight_pipeline.py': {'queue': globalQ.graph, 'pid': os.getpid()}}

    # Start watcher then the plotter
    watcher(watch)
    pltt.start()

    FAME_flight(args.config,
                args.experiments,
                args.steps,
                args.use_existing,
                args.log_name,
                args.log_folder,
                args.cores,
                args.predict_model,
                args.space_mode,
                args.batch_outdir,
                priority_bin=args.priority_bin,
                manifest_metadata_file=args.manifest_metadata_file,
                manifest_metadata=args.manifest_metadata,
                note=args.note,
                kill_file=args.kill_file)

    try:
        ram_mean, ram_max = pltt.stop()
        logging.info(f'Average RAM:{ram_mean:.2f}GB, Max RAM:{ram_max:.2f}GB')
    except:
        logging.error("Memory tracker failed to shut down correctly.")

    logging.info("======= Done =======")

if __name__ == "__main__":
    main()
