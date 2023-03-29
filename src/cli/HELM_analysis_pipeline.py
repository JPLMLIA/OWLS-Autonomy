'''
Command line interface to the HELM data processing pipeline
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
import json
import errno
from pathlib import Path
import string
import copy
import shutil

from utils                         import logger
from utils.manifest                import AsdpManifest, load_manifest_metadata
from utils.memory_tracker.plotter  import Plotter, watcher
from fsw.HELM_FAME                 import validate
from fsw.HELM_FAME                 import utils
from fsw.HELM_FAME                 import preproc
from utils.dir_helper              import get_batch_subdir, get_exp_subdir
from fsw.HELM_FAME.LAP_tracker     import run_tracker
from fsw.HELM_FAME.tracker         import run_tracker as run_proj_tracker
from gsw.HELM_FAME.point_metrics   import run_point_evaluation
from gsw.HELM_FAME.track_metrics   import run_track_evaluation
from gsw.HELM_FAME.reporting       import aggregate_statistics
from fsw.HELM_FAME.features        import get_features
from fsw.HELM_FAME.classifier      import train, predict, predict_batch_metrics
from fsw.HELM_FAME.asdp            import mugshots, generate_SUEs, generate_DDs, rehydrate_mugshots
from gsw.HELM_FAME.visualizer      import visualization
from utils.pipelines               import get_override_config, pipeline_run_step

PREPROC_STEP = "preproc"
VALIDATE_STEP = "validate"
TRACKER_STEP = "tracker"
PROJ_TRACKER_STEP = "proj_tracker"
POINT_EVAL_STEP = "point_evaluation"
TRACK_EVAL_STEP = "track_evaluation"
FEATURES_STEP = "features"
TRAIN_STEP = "train"
PREDICT_STEP = "predict"
ASDP_STEP = "asdp"
MANIFEST_STEP = "manifest"
PIPELINE_TRAIN = "pipeline_train"
PIPELINE_PREDICT = "pipeline_predict"
PIPELINE_TRACKER_EVAL = "pipeline_tracker_eval"
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
                                 instrument="HELM")
    validate.validate_data_ground(exp_dir=experiment,
                                 holo_fpaths=files,
                                 preproc_fpaths=preproc_files,
                                 n_workers=config['_cores'],
                                 config=config,
                                 instrument="HELM")

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

def point_eval_experiment(experiment, config):
    '''Create/serialize point evaluation report and return as well'''
    experiment_name = Path(experiment).name
    eval_dir = get_exp_subdir('evaluation_dir', experiment, config)
    pe_score_report_fpath = op.join(eval_dir, experiment_name + '_point_evaluation_report.json')
    extended_report_fname = config['evaluation']['points']['by_track_report_file']
    if extended_report_fname is not None and len(extended_report_fname) > 0:
        extended_report_fpath = op.join(eval_dir, extended_report_fname)
    else:
        extended_report_fpath = None

    # Get true and proposed tracks
    label_csv_fpath = op.join(get_exp_subdir('label_dir', experiment, config),
                                f'{experiment_name}_labels.csv')
    if not op.exists(label_csv_fpath):
        logging.warning('No labels found for experiment {}. Skipping.'
                    .format(experiment))
        return None

    track_fpaths = sorted(glob.glob(op.join(
        get_exp_subdir('track_dir', experiment, config), '*.json')))

    # Run point evaluation. Results saved to `pe_score_report_fpath`
    return (experiment,
            run_point_evaluation(label_csv_fpath, track_fpaths, pe_score_report_fpath,
                                 extended_report_fpath, config))

def point_eval_load_cached(experiment, config):
    '''Deserialize point evaluation report from previous run'''
    experiment_name = Path(experiment).name
    pe_score_report_fpath = op.join(get_exp_subdir('evaluation_dir', experiment, config),
                                    experiment_name + '_point_evaluation_report.json')
    with open(pe_score_report_fpath) as jsonfile:
        return (experiment, json.load(jsonfile))

def point_eval_batch(scores, _, batch_outdir, config):
    '''Create point metrics histograms'''
    aggregate_statistics(data=scores,
                         metrics=config['evaluation']['points']['hist_metrics'],
                         n_bins=config['evaluation']['histogram_bins'],
                         outdir=get_batch_subdir('point_eval_dir', batch_outdir, config),
                         macro_metric_path=config['evaluation']['points']['means_score_report_file'],
                         metrics_raw_path=config['evaluation']['points']['raw_distributions_file'])

def track_eval_experiment(experiment, config):
    '''Evaluate tracks on experiment'''
    experiment_name = Path(experiment).name
    n_frames = len(validate.get_preprocs(experiment, config))
    # Get true and proposed tracks
    label_csv_fpath = op.join(get_exp_subdir('label_dir', experiment, config),
                                f'{experiment_name}_labels.csv')
    if not op.exists(label_csv_fpath):
        logging.warning("No labels csv for experiment {}. Skipping...".format(experiment))
        return None

    track_fpaths = sorted(glob.glob(op.join(get_exp_subdir('track_dir', experiment, config), '*.json')))

    # Run track evaluation. Results saved to `score_report_fpath`
    return (experiment,
            run_track_evaluation(label_csv_fpath, track_fpaths,
                                 get_exp_subdir('evaluation_dir', experiment, config),
                                 n_frames, experiment_name, config))

def track_eval_load_cached(experiment, config):
    '''Load existing track evaluations'''
    experiment_name = Path(experiment).name
    te_score_report_fpath = op.join(get_exp_subdir('evaluation_dir', experiment, config),
                                    experiment_name + '_track_evaluation_report.json')
    with open(te_score_report_fpath) as jsonfile:
        return (experiment, json.load(jsonfile))

def track_eval_batch(scores, _, batch_outdir, config):
    '''Create track metrics histograms'''
    aggregate_statistics(data=scores,
                         metrics=config['evaluation']['tracks']['hist_metrics'],
                         n_bins=config['evaluation']['histogram_bins'],
                         outdir=get_batch_subdir('track_eval_dir', batch_outdir, config),
                         micro_metric_path=config['evaluation']['tracks']['micro_score_report_file'],
                         macro_metric_path=config['evaluation']['tracks']['macro_score_report_file'])

def features_experiment(experiment, config):
    '''Compute features on experiment'''
    data_track_features = get_features(experiment=experiment,
                                       config=config,
                                       save=True,
                                       labeled=config['_train_feats'])
    if not data_track_features:
        logging.error(f'Could not extract features for experiment {experiment}')
        return

def train_batch(_, experiments, batch_outdir, config):
    '''Run training on batch of experiments'''
    train(experiments, batch_outdir, config)

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
    rehydrate_mugshots(experiment, config)
    visualization(experiment, config, "HELM", config['_cores'], cleanup=True)

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

    manifest = AsdpManifest('helm', priority_bin)
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
        'timestats_mean_intensity',
        'validate',
        op.join(validate_dir, exp_name + '_timestats_mean_intensity.csv'),
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

def parse_steps(step_names, use_existing, predict_model, space_mode, train_feats):
    '''Parses command line steps/pipeline keywords and returns list of steps to run.
       Step tuples include name of step, functions associated with step, and whether step can use existing products'''

    cache_allowed_steps = [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, POINT_EVAL_STEP, TRACK_EVAL_STEP]

    step_mappings = {
        PREPROC_STEP :      [preproc_experiment, None, None],
        VALIDATE_STEP :     [validate_experiment, validate_batch, None],
        PROJ_TRACKER_STEP : [proj_tracker_experiment, None, None],
        TRACKER_STEP :      [tracker_experiment, None, None],
        POINT_EVAL_STEP :   [point_eval_experiment, point_eval_batch, point_eval_load_cached],
        TRACK_EVAL_STEP :   [track_eval_experiment, track_eval_batch, track_eval_load_cached],
        FEATURES_STEP :     [features_experiment, None, None],
        TRAIN_STEP :        [None, train_batch, None],
        PREDICT_STEP :      [predict_experiment, predict_batch, None],
        ASDP_STEP :         [asdp_experiment, asdp_batch, None],
        MANIFEST_STEP :     [manifest_experiment, None, None]
    }

    pipelines = {
        PIPELINE_TRAIN :        [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, TRACK_EVAL_STEP, FEATURES_STEP, TRAIN_STEP],
        PIPELINE_PREDICT :      [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, FEATURES_STEP, PREDICT_STEP],
        PIPELINE_TRACKER_EVAL : [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, POINT_EVAL_STEP, TRACK_EVAL_STEP],
        PIPELINE_PRODUCTS :     [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, POINT_EVAL_STEP,
                                 TRACK_EVAL_STEP, FEATURES_STEP, PREDICT_STEP,
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

    if TRAIN_STEP in step_names and not train_feats:
        logging.error("--steps train requires --train_feats")

    if PREDICT_STEP in step_names and train_feats:
        logging.error("--steps predict shouldn't use --train_feats")


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

def HELM_analysis(config, experiments, steps, use_existing, log_name, log_folder, cores, predict_model, space_mode, batch_outdir, train_feats, priority_bin=0, manifest_metadata_file=None, manifest_metadata=None, note="", toga_config=None, kill_file=""):
    """Main logic function for the HELM analysis script"""

    global start_time
    start_time = timeit.default_timer()

    logger.setup_logger(log_name, log_folder)

    steps_to_run = parse_steps(steps, use_existing, predict_model, space_mode, train_feats)

    with open(config) as f:
        config = yaml.safe_load(f)

    if toga_config:
        with open(toga_config) as f:
            override_config = yaml.safe_load(f)
            config = get_override_config(config, override_config)

    logging.info("Loaded config.")

    manifest_metadata = load_manifest_metadata(manifest_metadata_file, manifest_metadata)

    # Mapping from each step to the exhaustive list of steps that should trigger a rerun
    exp_deps = {PREPROC_STEP : [],
                VALIDATE_STEP : [PREPROC_STEP],
                TRACKER_STEP : [VALIDATE_STEP, PREPROC_STEP],
                POINT_EVAL_STEP : [TRACKER_STEP],
                TRACK_EVAL_STEP : [TRACKER_STEP],
                FEATURES_STEP : [TRACKER_STEP]}

    # The expected output directories for each step; a step is rerun if any are empty
    exp_dirs = {PREPROC_STEP : ['preproc_dir'],
                VALIDATE_STEP : ['validate_dir'],
                TRACKER_STEP : ['track_dir', 'evaluation_dir'],
                PROJ_TRACKER_STEP : ['track_dir', 'evaluation_dir'],
                POINT_EVAL_STEP : ['evaluation_dir'],
                TRACK_EVAL_STEP : ['evaluation_dir'],
                PREDICT_STEP : ['predict_dir'],
                FEATURES_STEP : ['features_dir'],
                ASDP_STEP : ['asdp_dir']}

    exp_paths = {POINT_EVAL_STEP: [('evaluation_dir', '*_point_evaluation_report.json')],
                 TRACK_EVAL_STEP: [('evaluation_dir', '*_track_evaluation_report.json')]}

    # To keep pipeline step calling convention simpler, add one-off args to config here
    config['_cores'] = cores
    config['_model_absolute_path'] = predict_model
    config['_train_feats'] = train_feats
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

    parser.add_argument('--config',             default=op.join(op.abspath(op.dirname(__file__)), "configs", "helm_config.yml"),
                                                help="Path to configuration file. Default is cli/configs/helm_config.yml")

    parser.add_argument('--toga_config',        default="",
                                                help="Override subset of config with path to toga generated config")

    parser.add_argument('--experiments',        nargs='+',
                                                required=True,
                                                help="Glob-able string patterns indicating sets of data files to be processed.")

    all_steps = [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, PROJ_TRACKER_STEP, POINT_EVAL_STEP, TRACK_EVAL_STEP, FEATURES_STEP, TRAIN_STEP, PREDICT_STEP, ASDP_STEP, MANIFEST_STEP]
    pipeline_keywords = [PIPELINE_TRAIN, PIPELINE_PREDICT, PIPELINE_TRACKER_EVAL, PIPELINE_PRODUCTS, PIPELINE_SPACE]
    steps_options = all_steps + pipeline_keywords
    cache_allowed_steps = [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, POINT_EVAL_STEP, TRACK_EVAL_STEP]

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

    parser.add_argument('--log_name',           default="HELM_analysis_pipeline.log",
                                                help="Filename for the pipeline log. Default is HELM_analysis_pipeline.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    parser.add_argument('--train_feats',        action='store_true',
                                                help="Only load tracks matched with hand labels (e.g., for ML training)" )

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
    pltt = Plotter(save_to=op.join(args.log_folder, "HELM_analysis_pipeline_memory.mp4"))
    globalQ = pltt.get_queues('HELM_analysis_pipeline.py')

    # Set up the watcher arguments
    watch = {'HELM_analysis_pipeline.py': {'queue': globalQ.graph, 'pid': os.getpid()}}

    # Start watcher then the plotter
    watcher(watch)
    pltt.start()

    HELM_analysis(args.config,
                  args.experiments,
                  args.steps,
                  args.use_existing,
                  args.log_name,
                  args.log_folder,
                  args.cores,
                  args.predict_model,
                  args.space_mode,
                  args.batch_outdir,
                  args.train_feats,
                  priority_bin=args.priority_bin,
                  manifest_metadata_file=args.manifest_metadata_file,
                  manifest_metadata=args.manifest_metadata,
                  note=args.note,
                  toga_config=args.toga_config)

    try:
        ram_mean, ram_max = pltt.stop()
        logging.info(f'Average RAM:{ram_mean:.2f}GB, Max RAM:{ram_max:.2f}GB')
    except:
        logging.error("Memory tracker failed to shut down correctly.")

    logging.info("======= Done =======")


if __name__ == "__main__":
    main()
