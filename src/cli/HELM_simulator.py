'''
Command line interface for the HELM data simulator
'''
import sys
import os.path as op
import argparse
import logging
from datetime import datetime

import yaml

from helm_dhm.validate import utils
from helm_dhm.simulator.sim_tracks import run_track_sim
from helm_dhm.simulator.sim_holograms import run_hologram_sim
from helm_dhm.simulator.utils import config_check

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_sim_exp_name(config):
    """Helper to create a simulator directory name using datetime and config"""
    time = datetime.now().strftime('%Y%m%d_%H%M%S')

    n_nonmot = config['exp_params']['n_non_motile']
    n_mot = config['exp_params']['n_motile']

    return f'{time}_sim_max{n_mot}_motile_max{n_nonmot}_nonmotile'


if __name__ == '__main__':

    ###################################
    # Argument parsing

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default="configs/helm_config.yml", type=str,
                        nargs='+', required=True,
                        help="Glob-able path(s) to configuration file(s)")
    # TODO: help for below spits out all 100 possible values
    parser.add_argument('--n_exp', type=int, default=1, required=True,
                        choices=range(1, 100),
                        help="Number of experiments to create per config.")
    parser.add_argument('--sim_outdir', type=str, required=True,
                        help="Directory to save simulated experiments to. Will overwrite existing directory.")
    args = parser.parse_args()

    ###################################
    # Load simulation configurations

    if not args.configs:
        logger.warning('No config files found, exiting.')
        sys.exit(0)

    # Confirm creation of simulation directory
    utils._check_create_delete_dir(args.sim_outdir, overwrite=True)

    ###################################
    # Load/check configurations
    exp_configs = []
    for config_fpath in args.configs:
        with open(config_fpath, 'r') as yaml_f:
            config = yaml.safe_load(yaml_f)
            config_check(config)
            exp_configs.append(config)

    ###################################
    # Simulate `n_exp` experiments per config
    logger.info(f'Starting simulation of {len(exp_configs) * args.n_exp} total experiments.')

    for config in exp_configs:
        for ci in range(args.n_exp):
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

            logger.info(f'\nStarting simulation of experiment: {exp_dir}')
            ###########################
            # Create tracks
            run_track_sim(config, exp_dir)

            ###########################
            # Create holograms
            run_hologram_sim(config, exp_dir)
