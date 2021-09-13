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

from utils                         import logger
from utils.manifest                import AsdpManifest, load_manifest_metadata
from utils.memory_tracker.plotter  import Plotter, watcher

from helm_dhm.validate import process
from helm_dhm.validate import utils
from helm_dhm.validate import preproc
from utils.dir_helper  import get_batch_subdir, get_exp_subdir

from helm_dhm.tracker.LAP_tracker           import run_tracker
from helm_dhm.tracker.tracker               import run_tracker as run_proj_tracker
from helm_dhm.evaluation.point_metrics      import run_point_evaluation
from helm_dhm.evaluation.track_metrics      import run_track_evaluation
from helm_dhm.evaluation.reporting          import aggregate_statistics
from helm_dhm.features.features             import get_features
from helm_dhm.classifier.classifier         import train, predict, predict_batch_metrics
from helm_dhm.asdp.asdp                     import mugshots, generate_SUEs, generate_DDs
from tools.visualizer.render                import visualization

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
PIPELINE_FIELD = "pipeline_field"

### Pipeline Steps ###

def preproc_experiment(experiment, config):
    '''Preprocess hologram files'''
    files = process.get_files(experiment, config)
    preproc.resize_holograms(holo_fpaths=files,
                             outdir=get_exp_subdir('preproc_dir', experiment, config, rm_existing=True),
                             resize_shape=config['preproc_resolution'],
                             n_workers=config['_cores'])

def validate_experiment(experiment, config):
    '''Create per experiment validate products'''
    files = process.get_files(experiment, config)
    preproc_files = process.get_preprocs(experiment, config)
    process.validate_data(exp_dir=experiment,
                            holo_fpaths=files,
                            preproc_fpaths=preproc_files,
                            n_workers=config['_cores'],
                            config=config,
                            instrument="HELM")

def validate_batch(_, experiments, batch_outdir, config):
    '''Calculate global statistics'''
    process.global_stats(exp_dirs=experiments,
                            out_dir=batch_outdir,
                            config=config)

def proj_tracker_experiment(experiment, config):
    '''Run the tracker on experiment'''
    files = process.get_files(experiment, config)
    preproc_files = process.get_preprocs(experiment, config)
    run_proj_tracker(exp_dir=experiment,
                holograms=preproc_files,
                originals=files,
                config=config,
                n_workers=config['_cores'])

def tracker_experiment(experiment, config):
    '''Run the tracker on experiment'''
    files = process.get_files(experiment, config)
    preproc_files = process.get_preprocs(experiment, config)
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
    n_frames = len(process.get_preprocs(experiment, config))
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
    predict_dir = get_exp_subdir('predict_dir', experiment, config)
    track_fpaths = sorted(glob.glob(op.join(predict_dir, '*.json')))
    holograms = process.get_files(experiment, config)
    num_files = len(holograms)

    mugshots(experiment, holograms, experiment, os.path.join(asdp_dir,"mugshots"), config)
    generate_SUEs(experiment, asdp_dir, track_fpaths, config['sue'])
    generate_DDs(experiment, asdp_dir, track_fpaths, config['dd'])

    if not config['_field_mode']:
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
        op.join(validate_dir, exp_name + '_mhi.png'),
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

    manifest.write(op.join(asdp_dir, exp_name + '_manifest.json'))


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
    experiment_directories = {PREPROC_STEP : ['preproc_dir'],
                              VALIDATE_STEP : ['validate_dir'],
                              TRACKER_STEP : ['track_dir', 'evaluation_dir'],
                              PROJ_TRACKER_STEP : ['track_dir', 'evaluation_dir'],
                              POINT_EVAL_STEP : ['evaluation_dir'],
                              TRACK_EVAL_STEP : ['evaluation_dir'],
                              PREDICT_STEP : ['predict_dir'],
                              FEATURES_STEP : ['features_dir'],
                              ASDP_STEP : ['asdp_dir']}

    for directory in experiment_directories[step]:
        exp_dir = get_exp_subdir(directory, experiment, config)
        if not op.isdir(exp_dir) or len(os.listdir(exp_dir)) == 0:
            logging.warning("\tStep {} does not have output {} at {}!".format(step, directory, exp_dir))
            return False

    # Additional per step files here
    paths = []
    experiment_name = Path(experiment).name
    if step == POINT_EVAL_STEP:
        paths.append(op.join(get_exp_subdir('evaluation_dir', experiment, config),
                                experiment_name + '_point_evaluation_report.json'))
    elif step == TRACK_EVAL_STEP:
        paths.append(op.join(get_exp_subdir('evaluation_dir', experiment, config),
                                experiment_name + '_track_evaluation_report.json'))
    for path in paths:
        if not op.exists(path):
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
    experiment_dependencies = {PREPROC_STEP : [],
                               VALIDATE_STEP : [PREPROC_STEP],
                               TRACKER_STEP : [VALIDATE_STEP, PREPROC_STEP],
                               POINT_EVAL_STEP : [TRACKER_STEP],
                               TRACK_EVAL_STEP : [TRACKER_STEP],
                               FEATURES_STEP : [TRACKER_STEP]}

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
    predict_model = cli_args.predict_model
    train_feats = cli_args.train_feats
    field_mode = cli_args.field_mode

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
        PIPELINE_FIELD :        [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, FEATURES_STEP,
                                 PREDICT_STEP, ASDP_STEP, MANIFEST_STEP]
    }

    # Pipeline-wise check
    if PIPELINE_FIELD in step_names and not field_mode:
        logging.error("--steps pipeline_field requires --field_mode")

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

    logging.info("\x1b[1mBeginning {} step...\x1b[0m".format(step))  # Bold font
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

    parser.add_argument('--config',             default=op.join(op.abspath(op.dirname(__file__)), "configs", "helm_config_latest.yml"),
                                                help="Path to configuration file. Default is cli/configs/helm_config_latest.yml")

    parser.add_argument('--toga_config',        default="",
                                                help="Override subset of config with path to toga generated config")

    parser.add_argument('--experiments',        nargs='+',
                                                required=True,
                                                help="Glob-able string patterns indicating sets of data files to be processed.")

    all_steps = [PREPROC_STEP, VALIDATE_STEP, TRACKER_STEP, PROJ_TRACKER_STEP, POINT_EVAL_STEP, TRACK_EVAL_STEP, FEATURES_STEP, TRAIN_STEP, PREDICT_STEP, ASDP_STEP, MANIFEST_STEP]
    pipeline_keywords = [PIPELINE_TRAIN, PIPELINE_PREDICT, PIPELINE_TRACKER_EVAL, PIPELINE_PRODUCTS, PIPELINE_FIELD]
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

    parser.add_argument('--log_name',           default="HELM_pipeline.log",
                                                help="Filename for the pipeline log. Default is HELM_pipeline.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    parser.add_argument('--train_feats',        action='store_true',
                                                help="Only load tracks matched with hand labels (e.g., for ML training)" )

    parser.add_argument('--predict_model',      default=op.join(op.abspath(op.dirname(__file__)), "models", "classifier_labelbox_v01.pickle"),
                                                help="Path to the pretrained model for prediction. Default is models/classifier_labelbox_v01.pickle")

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

    if args.toga_config:
        with open(args.toga_config) as f:
            override_config = yaml.safe_load(f)
            config = get_override_config(config, override_config)

    logging.info("Loaded config.")

    manifest_metadata = load_manifest_metadata(
        args.manifest_metadata_file, args.manifest_metadata)

    # To keep pipeline step calling convention simpler, add one-off args to config here
    config['_cores'] = args.cores
    config['_model_absolute_path'] = args.predict_model
    config['_train_feats'] = args.train_feats
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
    pltt = Plotter(save_to=op.join(args.log_folder, "HELM_pipeline_memory.mp4"))
    globalQ = pltt.get_queues('HELM_pipeline.py')

    # Set up the watcher arguments
    watch = {'HELM_pipeline.py': {'queue': globalQ.graph, 'pid': os.getpid()}}

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
        logging.error("No experiments found!")

    run_time = timeit.default_timer() - start_time

    try:
        ram_mean, ram_max = pltt.stop()
        logging.info(f'Average RAM:{ram_mean:.2f}GB, Max RAM:{ram_max:.2f}GB')
    except:
        logging.error("Memory tracker failed to shut down correctly.")

    logging.info("Full script run time: {time:.1f} seconds".format(time=run_time))
    logging.info("======= Done =======")

if __name__ == "__main__":
    main()
