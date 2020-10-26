import os
import os.path as op
import pytest
import copy

import yaml

from helm_dhm.classifier.classifier import *
from helm_dhm.validate import utils
from utils.dir_helper import get_batch_subdir, get_exp_subdir
from sklearn.ensemble import RandomForestClassifier

def setup_module():
    """Initialize output directories"""
    
    # This directory should not be used by any other test file
    batch_outdir = "helm_dhm/classifier/test/test_batch"
    utils._check_create_delete_dir(batch_outdir, overwrite=True)


def test_write_metrics():
    """Tests if write_metrics() generates all of the expected files

    We don't really need to test for malformed inputs here, that's more for
    train() or predict() to handle, by the time it gets here it should be clean

    Just confirm that the files we expect are still being generated
    """

    with open("cli/configs/helm_config.yml", 'r') as f:
        config = yaml.safe_load(f)

    batch_outdir = "helm_dhm/classifier/test/test_batch"

    true_Y = ["motile", "motile", "non-motile", "non-motile"]
    pred_Y = ["motile", "non-motile", "motile", "non-motile"]
    prob_Y = [0.9, 0.8, 0.7, 0.6]

    write_metrics(true_Y, pred_Y, prob_Y, batch_outdir, config)

    output_dir = get_batch_subdir("classifier_dir", batch_outdir, config)
    assert op.isfile(op.join(output_dir, "report.txt"))
    assert op.isfile(op.join(output_dir, "roc_plot.png"))
    assert op.isfile(op.join(output_dir, "pr_plot.png"))
    assert op.isfile(op.join(output_dir, "confusion.png"))


def test_cross_validate():
    """Tests if cross_validate() runs through or throws errors appropriately
    """

    with open("cli/configs/helm_config.yml", 'r') as f:
        config = yaml.safe_load(f)

    batch_outdir = "helm_dhm/classifier/test/test_batch"
    output_dir = get_batch_subdir("classifier_dir", batch_outdir, config)

    clf = RandomForestClassifier()

    X = [[1, 2, 3],
         [3, 4, 5],
         [6, 7, 8],
         [9, 10, 11]]
    Y = ['motile', 'motile', 'non-motile', 'non-motile']
    groups = [1, 1, 2, 2]
    
    ### 1. Invalid number of folds specified
    #   This should cause cross_val to skip and not throw any exceptions

    config['classifier']['cv_folds'] = 1
    cross_validate(clf, X, Y, groups, batch_outdir, config)

    assert not op.isfile(op.join(output_dir, "crossval_report.txt"))
    assert not op.isfile(op.join(output_dir, "crossval_roc_plot.png"))

    ### 2. Cannot do # of folds given number of groups
    #   This should cause cross_val to skip and not throw any exceptions

    config['classifier']['cv_folds'] = 2
    groups = [1, 1, 1, 1]
    cross_validate(clf, X, Y, groups, batch_outdir, config)

    assert not op.isfile(op.join(output_dir, "crossval_report.txt"))
    assert not op.isfile(op.join(output_dir, "crossval_roc_plot.png"))

    ### 3. Successful crossvalidation all the way through
    #   This should generate reports and figures
    groups = [1, 1, 2, 2]
    cross_validate(clf, X, Y, groups, batch_outdir, config)

    assert op.isfile(op.join(output_dir, "crossval_report.txt"))
    assert op.isfile(op.join(output_dir, "crossval_roc_plot.png"))

def test_train_predict():
    """Tests if train() and predict() runs through without throwing
    """

    with open("cli/configs/helm_config.yml", 'r') as f:
        config = yaml.safe_load(f)
    config['classifier']['do_cross_validation'] = False

    batch_outdir = "helm_dhm/classifier/test/test_batch"
    output_dir = get_batch_subdir("classifier_dir", batch_outdir, config)

    exps = ['helm_dhm/classifier/test/test_exps/exp1', 'helm_dhm/classifier/test/test_exps/exp2', 'helm_dhm/classifier/test/test_exps/exp3']
    empty_exp = ['helm_dhm/classifier/test/test_exps/exp_empty']


    ### 1. Empty experiment, no tracks, no features
      # This doesn't throw an exception, but it also doesn't output model or metrics
      # We don't test the metrics output here, different test already does that
    train(empty_exp, batch_outdir, config)
    assert not op.isfile(op.join(output_dir, config['classifier']['model_savepath']))

    ### 2. Experiments, but all features are masked out, no rows remain
    temp_config = copy.deepcopy(config)
    for k in temp_config['features']['mask'].keys():
        temp_config['features']['mask'][k] = 0
    train(exps, batch_outdir, temp_config)
    assert not op.isfile(op.join(output_dir, config['classifier']['model_savepath']))

    ### 3. Experiments, correct run through
    train(exps, batch_outdir, config)
    assert op.isfile(op.join(output_dir, config['classifier']['model_savepath']))

    config['_model_absolute_path'] = op.join(output_dir, config['classifier']['model_savepath'])
    ### 4. Predict on empty experiment
    assert predict(empty_exp[0], config) is None

    ### 5. Predict run through
    for e in exps:
        assert predict(e, config) is not None
