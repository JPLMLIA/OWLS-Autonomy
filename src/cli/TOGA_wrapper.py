'''
Command line interface for running TOGA on the HELM pipeline
'''
import sys
import os
import argparse
import logging
import timeit
import subprocess
import tempfile
import yaml
import json
from glob import glob

import os.path as op

sys.path.append(op.dirname(op.dirname(op.abspath(__file__))))
from utils import logger

def write_metrics(names, values, config):
    '''Write out the resulting HELM metrics to file expected by TOGA'''

    names = [str(n) for n in names]
    values = [str(v) for v in values]

    work_dir = op.join(os.getcwd(), config['output'])
    if not op.exists(work_dir):
        os.mkdir(work_dir)
    output = op.join(work_dir, config['metrics'])

    with open(output, 'w') as outfile:
        outfile.write('{}\n{}'.format(','.join(names), ','.join(values)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # The TOGA config file
    parser.add_argument('--config', '-c',       required=True,
                                                help="Path to custom configuration")

    parser.add_argument('--experiment_dir',     required=True,
                                                help="Path to experiment")

    parser.add_argument('--log_name',           default="TOGA_wrapper.log",
                                                help="Filename for the pipeline log")

    parser.add_argument('--log_folder',         default=op.join(op.abspath(op.dirname(__file__)), "logs"),
                                                help="Folder path to store logs. Default is cli/logs")

    args = parser.parse_args()

    logger.setup_logger(args.log_name, args.log_folder)

    if not 'PYTHON' in os.environ:
        raise Exception("Environment variable `PYTHON` must be set.")

    # Load the toga created config
    with open(args.config) as toga_yaml:
        config = yaml.load(toga_yaml, Loader=yaml.FullLoader)

    # Get dir containing helm pipeline
    cli_dir = op.dirname(op.abspath(__file__))

    # Override helm output dirs
    work_dir = op.join(os.getcwd(), config['output'])
    if not op.exists(work_dir):
        os.mkdir(work_dir)

    config['raw_batch_dir'] = True
    config['experiment_outputs_prefix'] = work_dir
    helm_batch_dir = op.join(work_dir, 'helm_batch_dir')

    # Dump the updated config for passing to helm
    tempf = tempfile.NamedTemporaryFile()
    with open(tempf.name, 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)

    cores = "1" # Limit to 1 core per worker, use multiple workers (see config on toga side)

    toga_start_time = timeit.default_timer()

    # Copy preproc directory and validate median
    exps = glob(args.experiment_dir)
    for exp in exps:
        validate_median = glob(op.join(exp, 'validate', '*_median_image.tif'))[0]
        prefix = op.join(config['experiment_outputs_prefix'], op.basename(exp))
        prefix_validate_dir = op.join(prefix, 'validate')
        prefix_validate_median = op.join(prefix, 'validate', op.basename(validate_median))
        subprocess.run(['mkdir', prefix])
        subprocess.run(['mkdir', prefix_validate_dir])
        subprocess.run(['cp', validate_median, prefix_validate_median])
        print(f'Copied prereqs to {prefix}')

    # Run the HELM pipeline with this config via subprocess
    # TODO: Add and remove point_evaluation and track_evaluation as necessary
    # in the steps option below
    subprocess.run(['nice',
                    '-n', '20',
                    os.environ['PYTHON'],
                    op.join(cli_dir, 'HELM_pipeline.py'),
                    '--experiments',  args.experiment_dir,
                    '--batch_outdir', helm_batch_dir,
                    '--config',       op.join(cli_dir, 'configs/helm_config.yml'),
                    '--toga_config',  tempf.name,
                    '--steps',        'tracker', 'track_evaluation',
                    '--cores',        cores,
                    '--space_mode'])

    # Parse HELM metrics and write TOGA compatible file
    # TODO: comment below lines in and out depending on whether optimizing on
    # point or track evaluation metrics (p_ and t_ prefixes, respectively).
    # Wrapper will break if looking for point or track metrics that don't exist.

    helm_config_path = op.join(cli_dir, "configs/helm_config.yml")

    with open(helm_config_path) as f:
        helm_config = yaml.safe_load(f)
    """
    p_score_fpath = op.join(helm_batch_dir,
                          helm_config['batch_dirs']['point_eval_dir'],
                          helm_config['evaluation']['points']['means_score_report_file'])
    """
    t_score_fpath = op.join(helm_batch_dir,
                          helm_config['batch_dirs']['track_eval_dir'],
                          helm_config['evaluation']['tracks']['macro_score_report_file'])

    """
    with open(p_score_fpath) as p_score_report:
        p_metrics = json.load(p_score_report)
    """
    with open(t_score_fpath) as t_score_report:
        t_metrics = json.load(t_score_report)

    metric_names = config['metric_names']
    """
    metric_values = [p_metrics[name] if name in p_metrics else t_metrics[name]
                     for name in metric_names]
    """
    metric_values = [t_metrics[name] for name in metric_names]

    write_metrics(metric_names, metric_values, config)

    toga_run_time = timeit.default_timer() - toga_start_time
    logging.info("TOGA_WRAPPER run time: {time:.1f} seconds".format(time=toga_run_time))
    logging.info("======= Done =======")
