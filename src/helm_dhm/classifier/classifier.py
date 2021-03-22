'''
HELM classifier library
'''
import os, sys
import os.path as op
import glob
import pickle
import logging
import csv
import json
from pathlib import Path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GroupKFold

from utils.dir_helper import get_batch_subdir, get_exp_subdir

IGNORE_FEATS = set(['motility', 'span', 'track', 'dataset_name'])

def write_metrics(true_Y, pred_Y, prob_Y, batch_outdir, config, prefix=""):
    """ Writes out classifier metrics. Restricted to binary classification.

    Only the labels "motile" and "non-motile" are expected, with "motile" as the
    positive label.

    Currently writes:
    - Classification Report
    - AUC curve plot
    - Precision-Recall curve plot
    - Confusion matrix

    Parameters
    ----------
    true_Y: list
        List of ground truth labels. In ["motile", "non-motile"].
    pred_Y: list
        List of predicted labels. In ["motile", "non-motile"].
    prob_Y: list
        List of "motile" probabilities.
    batch_outdir: string
        Batch output directory.
    config: dict
        Configuration read from YAML.
    prefix: str
        Prefix to be appended to the output filenames. Useful for specifying
        train vs test metric output.
        Defaults to "".
    """
    # Output directory path
    output_dir = get_batch_subdir("classifier_dir", batch_outdir, config)


    ### BASIC CLASSIFICATION REPORT
    report = metrics.classification_report(true_Y, pred_Y)
    if prefix != "":
        report_fp = op.join(output_dir, prefix+"_report.txt")
    else:
        report_fp = op.join(output_dir, "report.txt")

    # write to file
    with open(report_fp, 'w') as f:
        f.write("Classification Report with threshold {}\n".format(config['classifier']['motility_threshold']))
        f.write(report)

    logging.info(f'Saved motility classification report: {op.join(*Path(report_fp).parts[-2:])}')

    ### ROC PLOT
    fpr, tpr, _ = metrics.roc_curve(true_Y, prob_Y, pos_label="motile")
    # Binarize true labels to 1 for motile, 0 for non-motile
    binary_true_Y = [1 if x=='motile' else 0 for x in true_Y]
    # Calculate AUC
    auc = metrics.roc_auc_score(binary_true_Y, prob_Y)

    # Plot ROC curve
    fig, ax = plt.subplots(dpi=300)
    ax.plot(fpr, tpr, color="blue", label="ROC curve (area = {:.2f})".format(auc))
    ax.plot([0,1], [0,1], '--', color="red", label="Chance")
    ax.set_title("{} ROC Curve".format(prefix))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_aspect('equal')
    ax.legend()
    if prefix != "":
        fig.savefig(op.join(output_dir, "{}_roc_plot.png".format(prefix)))
    else:
        fig.savefig(op.join(output_dir, "roc_plot.png"))
    logging.info(f'Saved ROC plot: {op.join(*Path(op.join(output_dir, "*_roc_plot.png")).parts[-2:])}')


    ### PRECISION-RECALL PLOT
    precision, recall, _ = metrics.precision_recall_curve(true_Y, prob_Y, pos_label="motile")

    # Plot PR curve
    fig, ax = plt.subplots(dpi=300)
    ax.plot(recall, precision, color="blue")
    ax.set_title("{} Precision-Recall Curve".format(prefix))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_aspect('equal')
    if prefix != "":
        fig.savefig(op.join(output_dir, "{}_pr_plot.png".format(prefix)))
    else:
        fig.savefig(op.join(output_dir, "pr_plot.png"))
    logging.info(f'Saved prec-rec plot: {op.join(*Path(op.join(output_dir, "*_pr_plot.png")).parts[-2:])}')


    ### CONFUSION MATRIX
    confusion = metrics.confusion_matrix(true_Y, pred_Y, labels=['motile', 'non-motile'])

    # Plot confusion matrix
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(confusion, cmap='Blues')
    # x-axis formatting
    ax.set_xlabel("Predicted label")
    ax.set_xticks([0,1])
    ax.set_xticklabels(['motile', 'non-motile'])
    # y-axis formatting
    ax.set_ylabel("True label")
    ax.set_yticks([0,1])
    ax.set_yticklabels(['motile', 'non-motile'])
    # on-square text
    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion[i,j], ha='center', va='center', color='black')

    if prefix != "":
        fig.savefig(op.join(output_dir, "{}_confusion.png".format(prefix)))
    else:
        fig.savefig(op.join(output_dir, "confusion.png"))
    logging.info(f'Saved confusion matrix: {op.join(*Path(op.join(output_dir, "*_confusion.png")).parts[-2:])}')


def cross_validate(clf, X, Y, groups, batch_outdir, config):
    """ Performs k-fold cross validation on provided classifier

    Parameters
    ----------
    clf: sklearn classifier object
        Initialized classifier. Any existing learned parameters will be
        overwritten.
    X: numpy array
        Data and features to be trained on.
    Y: numpy array
        Labels to be trained on.
    group: numpy array
        Same value for tracks within the same experiment. For GroupKFold.
    batch_outdir: string
        Directory path to batch output directory
    config: dict
        Configuration read from YAML.

    Returns
    -------
    None.
    """

    ### Read from configuration
    # number of folds for cross validation
    cv_folds = config['classifier']['cv_folds']
    # directory for cross validation result output
    output_dir = get_batch_subdir('classifier_dir', batch_outdir, config)
    # probability threshold for labeling a track as motile
    threshold = config['classifier']['motility_threshold']


    ### Initialize k-fold stratified cross-validation
    try:
        # Using group k fold to avoid test/train within same exp
        skf = GroupKFold(n_splits=cv_folds)
    except Exception as e:
        logging.error("Failed to initialize cross validation, skipping:")
        logging.error(e)
        return

    ### Try splitting
    try:
        crossval_splits = skf.split(X, Y, groups)
        for _, (_, _) in enumerate(crossval_splits):
            pass
    except Exception as e:
        logging.error("Failed to split for cross validation, skipping:")
        logging.error(e)
        return

    ### Global AUC plot
    fig, ax = plt.subplots(dpi=300)
    ax.plot([0,1], [0,1], '--', color="red", label="Chance")
    ax.set_title("Crossval ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_aspect('equal')

    ### Global classification report output
    report_fp = op.join(output_dir, "crossval_report.txt")

    # write to file
    with open(report_fp, 'w') as f:
        f.write("Classification Report with threshold {}\n".format(threshold))

    for curr_fold, (train_index, test_index) in enumerate(crossval_splits):
        # For each split...
        train_X = X[train_index]
        train_Y = Y[train_index]
        test_X = X[test_index]
        test_Y = Y[test_index]

        # Train model
        clf.fit(train_X, train_Y)

        # Predict probabilities for AUC curve
        pred_Y = clf.predict_proba(test_X)
        pred_classes = clf.classes_

        # predict_proba() returns probs for both classes, find out which is motile
        motile_col = np.where(pred_classes == 'motile')[0][0]
        pred_Y = pred_Y[:,motile_col]

        # Use configured threshold to assign labels 'motile' and 'non-motile'
        num_tracks = len(pred_Y)
        pred_Y_labels = np.array(['non-motile'] * num_tracks, dtype=object)
        pred_Y_labels[pred_Y > threshold] = 'motile'
        binary_test_Y = [1 if x=='motile' else 0 for x in test_Y]

        # Write to reports
        report = metrics.classification_report(test_Y, pred_Y_labels)
        with open(report_fp, 'a') as f:
            f.write("\n")
            f.write("Fold {}".format(curr_fold))
            f.write(report)

        # Calculate ROC and AUC and add to plot
        fpr, tpr, _ = metrics.roc_curve(test_Y, pred_Y, pos_label='motile')
        auc = metrics.roc_auc_score(binary_test_Y, pred_Y)

        ax.plot(fpr, tpr, label="Fold {0} (area = {1:.2f})".format(curr_fold, auc))

    ax.legend()
    fig.savefig(op.join(output_dir, "crossval_roc_plot.png"))

def train(experiments, batch_outdir, config, hyperparams={"max_depth": 10}):
    """ Trains an sklearn random forest model on input features and saves it as a pickle

    Parameters
    ----------
    experiments: list
        List of experiments generated by pipeline-level glob
    batch_outdir: string
        Output directory for batch-level metrics and trained model
    config: dict
        Configuration dictionary read in by pipeline from YAML
    hyperparams: dict
        Hyperparameters for model training. Exposed for DOMINE optimization.
        NOTE: Temporarily defaults to {"max_depth": 5}
        NOTE: Do not add hyperparameters to config, as it will be fixed eventually

    Returns
    -------
    None
    TODO: Return metrics for DOMINE optimization
    """

    # Batch-level feature and label storage
    batch_X = []
    batch_Y = []
    groups = []
    feat_columns = None
    for group_id, exp in enumerate(experiments):
        # Get feature CSV filepath
        feat_subdir = get_exp_subdir('features_dir', exp, config)
        feat_filepath = op.join(feat_subdir, config['features']['output'])

        # Read in feature CSV
        with open(feat_filepath, 'r') as f:
            reader = csv.DictReader(f)

            if feat_columns is None:
                feat_columns = [feat for feat in reader.fieldnames if (feat not in IGNORE_FEATS)]

            for row in reader:
                # Assert that the motility column exists
                if 'motility' not in row.keys():
                    # break to catch the empty dataset
                    break

                # Add label to label set
                batch_Y.append(row['motility'].lower())

                # Add features to feature set
                batch_X.append([row[feat] for feat in feat_columns])

                # Record group for cross-validation
                groups.append(group_id)

    batch_X = np.array(batch_X).astype(np.float32)
    batch_Y = np.array(batch_Y, dtype=object)
    groups = np.array(groups)

    if not batch_X.size:
        logging.error("No valid rows found in features file, exiting training without output.")
        return

    ### FILTER LABELS
    ### TODO: Decide what to do with "Ambiguous" or other labels
    ### Currently only "Motile" and "Non-Motile" are kept. 07/29/2020 JL

    keep_indices = []                           # indices to keep
    found_nonlabels = set()                     # record found bad labels
    drop_count = 0                              # number of tracks filtered out

    # Build binary mask of rows with non-standard labels for deletion
    for i in range(len(batch_X)):
        if batch_Y[i].lower() not in ['motile', 'non-motile']:
            found_nonlabels.add(batch_Y[i])
            drop_count += 1
        else:
            keep_indices.append(i)

    # Don't train on any tracks that aren't Motile or Non-motile
    if drop_count:
        logging.warning("Non-standard labels encountered: {}".format(found_nonlabels))
        logging.warning("{} tracks dropped from training.".format(drop_count))
        # This uses the binary mask to only keep rows where the mask val is 1
        batch_X = batch_X[keep_indices]
        batch_Y = batch_Y[keep_indices]
        groups = groups[keep_indices]

    if not batch_X.size:
        logging.error("No tracks remain after label filtering, exiting training without output.")
        return

    ### PREPROCESS OR AUGMENT
    ### TODO: At some point, if we use anything other than decision trees, we'll
    ### need to standardize features or something. Do that here, and consider
    ### writing helper functions.

    # replacing infinite features with numbers
    batch_X = np.nan_to_num(batch_X)


    ### INITIALIZE MODEL
    clf = RandomForestClassifier(**hyperparams)


    ### CROSS VALIDATION
    if config['classifier']['do_cross_validation']:
        logging.info('Cross validation enabled, running...')
        cross_validate(clf, batch_X, batch_Y, groups, batch_outdir, config)


    ### TRAIN MODEL ON ALL TRAINING DATA
    ### This occurs regardless of cross validation
    clf.fit(batch_X, batch_Y)


    ### SAVE MODEL TO SPECIFIED PATH
    class_dir = get_batch_subdir('classifier_dir', batch_outdir, config)
    model_savepath = op.join(class_dir, config['classifier']['model_savepath'])
    with open(model_savepath, 'wb') as f:
        pickle.dump((clf, feat_columns), f)
    logging.info(f'Saved trained model: {op.join(*Path(model_savepath).parts[-2:])}')


    ### SAVE METRICS

    # Predict probabilities for AUC curve and Precision-Recall curve
    pred_Y = clf.predict_proba(batch_X)
    pred_classes = clf.classes_

    # predict_proba() returns probs for both classes, find out which is motile
    motile_col = np.where(pred_classes == 'motile')[0][0]
    prob_Y = pred_Y[:,motile_col]

    # Use configured threshold to assign labels 'motile' and 'non-motile'
    threshold = config['classifier']['motility_threshold']
    num_tracks = len(prob_Y)
    pred_Y_labels = np.array(['non-motile'] * num_tracks, dtype=object)
    pred_Y_labels[prob_Y > threshold] = 'motile'

    # Write metrics
    write_metrics(batch_Y, pred_Y_labels, prob_Y, batch_outdir, config, "train")

def predict(experiment, config):
    """ Tests an sklearn model on input features and writes prediction JSONs

    Parameters
    ----------
    experiment: str
        The experiment to predict on
    config: dict
        Configuration dictionary read in by pipeline from YAML

    Returns
    -------
    None
    TODO: Return metrics for DOMINE optimization?
        This would be done by writing to a file via directory helpers.
        Toga will be able to override directory logic to obtain metrics.
    """

    model_path = config['_model_absolute_path']

    ### LOAD CLASSIFIER FROM PICKLE
    try:
        with open(model_path, 'rb') as f:
            clf, feat_columns = pickle.load(f)
        logging.info(f"Found and loaded {model_path}")
    except:
        logging.warning(f"Failed to open classifier {model_path}")
        return None

    # Storage for batch-level metrics
    batch_true_Y = []
    batch_pred_Y = []
    batch_prob_Y = []
    batch_alltracks = 0

    # Get feature CSV filepath
    feat_subdir = get_exp_subdir('features_dir', experiment, config)
    feat_filepath = op.join(feat_subdir, config['features']['output'])
    # Get track JSON directory
    track_subdir = get_exp_subdir('track_dir', experiment, config)
    # Get output predict directory
    predict_subdir = get_exp_subdir('predict_dir', experiment, config, rm_existing=True)

    if not os.path.exists(feat_filepath):
        logging.error(f"Feature file {feat_filepath} missing. Aborting classification.")
        return

    ### READ FEATURES FROM CSV FILE
    exp_X = []
    exp_Y = [] # labels are for metrics
    track_ID = []
    with open(feat_filepath, 'r') as f:
        reader = csv.DictReader(f)

        # Assert features aren't empty or no header
        if not reader.fieldnames:
            logging.error(f"Features are empty or lacks header row.")
            return None

        # Assert that its features list is the same as training
        this_keys = [feat for feat in reader.fieldnames if (feat not in IGNORE_FEATS)]
        if set(this_keys) != set(feat_columns):
            logging.error(f"Read features list {this_keys} doesn't match model's {feat_columns}")
            return None

        for row in reader:
            # Save labels if they exist
            if 'motility' not in row.keys():
                exp_Y.append('')
            else:
                exp_Y.append(row['motility'].lower())

            # Assemble features in the same order as training data
            exp_X.append([row[feat] for feat in feat_columns])

            track_ID.append(int(row['track']))

    exp_X = np.array(exp_X).astype(np.float32)
    exp_Y = np.array(exp_Y, dtype=object)

    if exp_X.size == 0:
        logging.error("No tracks found in directory.")
        return None

    ### PREPROCESS OR AUGMENT
    ### TODO: At some point, if we use anything other than decision trees, we'll
    ### need to standardize features or something. Do that here, and consider
    ### writing helper functions.

    # replacing infinite features with numbers
    exp_X = np.nan_to_num(exp_X)

    ### PREDICT
    pred_Y = clf.predict_proba(exp_X)

    # predict_proba() returns probs for both classes, find out which is motile
    pred_classes = clf.classes_
    motile_col = np.where(pred_classes == 'motile')[0][0]
    prob_Y = pred_Y[:,motile_col]

    # Use configured threshold to classify into 'motile' and 'other'
    # TODO: Using 'other' here for visualizer but 'non-motile' is probably better
    # Change 'other' to 'non-motile' in both classifier and visualizer
    threshold = config['classifier']['motility_threshold']
    num_tracks = len(prob_Y)
    pred_Y_labels = np.array(['other'] * num_tracks, dtype=object)
    pred_Y_labels[prob_Y > threshold] = 'motile'

    # Metrics writer expects 'motile' and 'non-motile'
    metrics_compat = np.array(['non-motile'] * num_tracks, dtype=object)
    metrics_compat[prob_Y > threshold] = 'motile'

    ### WRITE TO PREDICT TRACK JSONS
    track_fpaths = sorted(glob.glob(op.join(track_subdir, '*' + config['track']['ext'])))

    for i in range(num_tracks):
        # new keys and values to be added to the JSON
        update_dict = {'classification': pred_Y_labels[i],
                        'probability_motility': prob_Y[i]}

        # we're doing this just in case sequences are nonconsecutive
        with open(track_fpaths[track_ID[i]], 'r') as f:
            data = json.load(f)
        data.update(update_dict)

        # write out JSON files
        with open(op.join(predict_subdir, op.basename(track_fpaths[track_ID[i]])), 'w') as f:
            json.dump(data, f, indent=4)
    
    logging.info(f'Saved predictions: {op.join(*Path(predict_subdir).parts[-2:])}')

    ### IF TRACKS HAVE LABELS, ADD TO BATCH STORAGE FOR METRICS
    for i in range(num_tracks):
        if exp_Y[i].lower() in ['motile', 'non-motile']:
            # this track has a valid label
            batch_true_Y.append(exp_Y[i])
            batch_pred_Y.append(metrics_compat[i])
            batch_prob_Y.append(prob_Y[i])
    batch_alltracks += num_tracks

    return (batch_true_Y, batch_pred_Y, batch_prob_Y, num_tracks)

def predict_batch_metrics(batch_true_Y, batch_pred_Y, batch_prob_Y, batch_alltracks, batch_outdir, config):
    '''Calculate batch metrics if labels exist'''
    if len(batch_true_Y):
        logging.info("{} of {} tracks have labels, calculating batch metrics".format(
                        len(batch_true_Y), batch_alltracks))
        write_metrics(batch_true_Y, batch_pred_Y, batch_prob_Y, batch_outdir, config, "predict")
