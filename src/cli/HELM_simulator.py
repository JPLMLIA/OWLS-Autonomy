'''
Command line interface for the HELM data simulator
'''
import sys
import os.path as op
import argparse
import logging
from datetime import datetime
from glob import glob

import yaml

from helm_dhm.validate                import utils
from helm_dhm.simulator.sim_tracks    import run_track_sim
from helm_dhm.simulator.sim_holograms import run_hologram_sim
from helm_dhm.simulator.utils         import config_check
from utils                            import logger

def make_sim_exp_name(config):
    """Helper to create a simulator directory name using datetime and config"""
    time = datetime.now().strftime('%Y%m%d_%H%M%S')

    n_nonmot = config['exp_params']['n_non_motile']
    n_mot = config['exp_params']['n_motile']

    return f'{time}_sim_max{n_mot}_motile_max{n_nonmot}_nonmotile'


def main():

    ###################################
    # Argument parsing

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--configs',            default=op.join(op.abspath(op.dirname(__file__)), "configs", "helm_simulator_config.yml"),
                                                type=str,
                                                nargs='+',
                                                help="Glob-able path(s) to configuration file(s). Default is configs/helm_simulator_config.yml")

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
    config_paths = glob(args.configs)
    exp_configs = []
    for config_fpath in config_paths:
        with open(config_fpath, 'r') as yaml_f:
            config = yaml.safe_load(yaml_f)
            config_check(config)
            exp_configs.append(config)

    ###################################
    # Simulate `n_exp` experiments per config
    logging.info(f'Starting simulation of {len(exp_configs) * args.n_exp} total experiments.')

    for config in exp_configs:
        for _ in range(args.n_exp):
            ###########################
            # Setup

            # Determine the experiment directory and subdirs
            exp_name = make_sim_exp_name(config)
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
