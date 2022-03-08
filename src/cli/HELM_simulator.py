'''
Command line interface for the HELM data simulator
'''
import sys
import os.path as op
import argparse
import logging
from datetime import datetime
from glob import glob
from pathlib import Path

import yaml

from fsw.HELM_FAME                      import utils
from tools.helm.simulator.sim_tracks    import run_track_sim
from tools.helm.simulator.sim_holograms import run_hologram_sim
from tools.helm.simulator.utils         import config_check
from utils                              import logger

def make_sim_exp_names(config, n_exp):
    """Helper to create simulator directory names using datetime and config"""

    # Pull out some metadata
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    n_nonmot = config['exp_params']['n_non_motile']
    n_mot = config['exp_params']['n_motile']
    has_drift = config['exp_params'].get('drift') is not None

    exp_dirnames = []

    # Generate experiment names. One per repeat
    for ei in range(n_exp):
        flow_str = 'flow' if has_drift else 'static'
        exp_dirnames.append(f'{date}_dhm_{flow_str}_max{n_mot}_motile_max{n_nonmot}_nonmotile_grayscale_sim_{ei:02}')

    return exp_dirnames


def main():

    ###################################
    # Argument parsing

    parser = argparse.ArgumentParser()

    parser.add_argument('--configs',            default=op.join(op.abspath(op.dirname(__file__)), "configs", "helm_simulator_config_v2.yml"),
                                                type=str,
                                                nargs='+',
                                                help="Glob-able path(s) to configuration file(s). Default is configs/helm_simulator_config_v2.yml")

    # TODO: help for below spits out all 100 possible values
    parser.add_argument('--n_exp',              type=int,
                                                default=1,
                                                choices=range(1, 100),
                                                help="Number of experiments to create per config. Defaults to 1.")

    parser.add_argument('--sim_outdir',         type=str,
                                                required=True,
                                                help="Directory to save simulated experiments to. Will overwrite existing directory.")

    parser.add_argument('--log_name',           default="HELM_simulator.log",
                                                help="Filename for the pipeline log. Default is HELM_simulator.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()


    ###################################
    # Configure logging

    logger.setup_logger(args.log_name, args.log_folder)

    ###################################
    # Load simulation configurations

    if not args.configs:
        logging.warning('No config files found, exiting.')
        sys.exit(0)

    # Confirm creation of simulation directory
    utils._check_create_delete_dir(args.sim_outdir, overwrite=True)

    ###################################
    # Load/check configurations
    #config_paths = glob(args.configs)
    config_fpaths = set()
    for pattern in args.configs:
        curr_dirs = sorted([f for f in glob(pattern) if op.isfile(Path(f))])
        config_fpaths.update(curr_dirs)

    exp_configs = []
    for config_fpath in config_fpaths:
        with open(config_fpath, 'r') as yaml_f:
            config = yaml.safe_load(yaml_f)
            config_check(config.copy())
            exp_configs.append(config)

    ###################################
    # Simulate `n_exp` experiments per config
    logging.info(f'Starting simulation of {len(exp_configs) * args.n_exp} total experiments.')

    for config in exp_configs:
        exp_names = make_sim_exp_names(config, args.n_exp)
        for exp_name in exp_names:
            ###########################
            # Determine the experiment directory and subdirs
            exp_dir = op.join(args.sim_outdir, exp_name)
            sim_track_dir = op.join(args.sim_outdir, exp_name, config['sim_track_dir'])
            sim_hologram_dir = op.join(args.sim_outdir, exp_name, config['sim_hologram_dir'])

            # Create the experiment directory and subdirs
            utils._check_create_delete_dir(exp_dir, overwrite=True)
            utils._check_create_delete_dir(sim_track_dir, overwrite=True)
            utils._check_create_delete_dir(sim_hologram_dir, overwrite=True)

            logging.info(f'\nStarting simulation of experiment: {exp_dir}')
            ###########################
            # Create tracks
            run_track_sim(config, exp_dir)

            ###########################
            # Create holograms
            run_hologram_sim(config, exp_dir)
