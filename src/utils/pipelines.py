import os
import copy
import logging
import timeit

from datetime         import datetime
from utils.dir_helper import get_exp_subdir
from glob             import glob

import os.path as op

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

def has_experiment_outputs(step, experiment, exp_directories, exp_paths, config):
    '''Returns true if experiment already has all expected step outputs'''

    for directory in exp_directories[step]:
        exp_dir = get_exp_subdir(directory, experiment, config)
        if not op.isdir(exp_dir) or len(os.listdir(exp_dir)) == 0:
            logging.warning("\tStep {} does not have output {} at {}!".format(step, directory, exp_dir))
            return False

    # Additional per step files here
    if step in exp_paths:
        for path in exp_paths[step]:
            realpath = glob(op.join(get_exp_subdir(path[0], experiment, config), path[1]))[0]
            if not op.exists(realpath):
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

def should_run(step, use_preexisting, experiment, exp_deps, 
                exp_dirs, exp_paths, config):
    '''Determine if step needs to be rerun on experiment'''

    # Run if caching disabled
    if not use_preexisting:
        return True

    # Run if outputs don't exists, or were run with a different config
    if not has_experiment_outputs(step, experiment, exp_dirs, exp_paths, config):
        return True

    # TODO: Run if config doesn't match previous run

    # Rerun if any of of the steps depended on by this step were run more recently
    step_ts = get_timestamp(step, experiment, config)
    if not step_ts:
        return True
    for dependency in exp_deps[step]:
        dep_ts = get_timestamp(dependency, experiment, config)
        if not dep_ts or dep_ts > step_ts:
            return True

    return False

def pipeline_run_step(step_tuple, experiments, batch_outdir, exp_deps, exp_dirs, exp_paths, config):
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
    exp_deps : dict
        Each step's output dependencies
    exp_dirs : dict
        Each step's expected output directories
    exp_dirs : dict
        Each step's expected output files, in addition to dirs
    config : dict 
        The config for this run
    """

    # Parse step tuple
    step = step_tuple[0]
    experiment_func = step_tuple[1]
    batch_func = step_tuple[2]
    get_preexisting_func = step_tuple[3]
    use_preexisting =  step_tuple[4]

    if os.path.exists(config['_kill_file']):
        logging.error(f"Encountered kill file {config['_kill_file']} on step {step}. Halting operations. Remove file to re-enable pipeline usage")
    else:
        logging.info("\x1b[1mBeginning {} step...\x1b[0m".format(step))  # Bold font
        st = timeit.default_timer()
        outputs = []
        # Run per experiment steps (if any)
        if experiment_func:
            for experiment in experiments:
                # Skip running on an experiment if we can use a pre-existing result
                if not should_run(step, use_preexisting, experiment, exp_deps, exp_dirs, exp_paths, config):
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
