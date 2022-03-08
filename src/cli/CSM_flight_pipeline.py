'''
Command line interface to the CSM autonomy pipeline.
'''

import os.path as op
import logging
import argparse
import timeit
import yaml
import csv
import numpy as np

from utils import logger

def CSM_flight(config, compath, log_name, log_folder):
    """CSM Autonomy Function

    Args:
        config (String): Path to configuration YAML
        compath (String): Path to COM CSV
        log_name (String): Name of log file
        log_folder (String): Name of log directory

    Returns:
        int: 0 for low, 1 for med, and 2 for high ionic strength.
    """

    # initialize
    global start_time
    start_time = timeit.default_timer()
    logger.setup_logger(log_name, log_folder)

    # load config
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    # load data from COM
    data = []    
    with open(compath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if row[3] == "bp_CSM.EC_Conductivity":
                data.append(float(row[4]))
    
    EC_avg = np.mean(data)
    EC_std = np.std(data)
    logging.info(f"EC avg: {EC_avg:e} std: {EC_std:e}")

    # check coefficient of variation
    if EC_std/EC_avg > config['cv_thresh']:
        logging.warning(f"EC CV {EC_std/EC_avg:.2f} greater than thresh {config['cv_thresh']}")
    
    # calculate ionic strength
    IS_avg = (EC_avg / config['expmodel']['coeff']) ** (1.0 / config['expmodel']['exp'])

    # thresholds for int return
    code = -1
    if IS_avg > config['thresholds']['high']:
        logging.info(f"IS avg: {IS_avg:e} class: 2 (high)")
        code = 2
    elif IS_avg > config['thresholds']['med']:
        logging.info(f"IS avg: {IS_avg:e} class: 1 (med)")
        code = 1
    else:
        logging.info(f"IS avg: {IS_avg:e} class: 0 (low)")
        code = 0
    
    # runtime
    run_time = timeit.default_timer() - start_time
    logging.info("Full script run time: {time:.1f} seconds".format(time=run_time))

    # logging cleanup
    for x in range(0, len(logging.getLogger().handlers)):
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])

    return code

    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('compath',              help="Path to COM CSV.")

    parser.add_argument('--config',             default=op.join(op.abspath(op.dirname(__file__)), "configs", "csm_config.yml"),
                                                help="Path to configuration file. Default is cli/configs/csm_config.yml")

    parser.add_argument('--log_name',           default="CSM_flight_pipeline.log",
                                                help="Filename for the pipeline log. Default is CSM_flight_pipeline.log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    CSM_flight(args.config, args.compath, args.log_name, args.log_folder)

if __name__ == "__main__":
    main()