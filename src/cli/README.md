# Command Line Interface Documentation

> Please double check that you've followed the installation instructions in the
  root README. The procedure has changed recently.
  This document assumes that you have run `pip install -e .` as instructed.

**Table of Contents**

- [ACME](#ACME)
  - [ACME Pipeline](#ACME_pipeline)
  - [ACME Simulator](#ACME_simulator)
  - [ACME evaluation](#ACME_evaluation)
- [HELM](#HELM)
  - [HELM Pipeline](#HELM_pipeline)
  - [HELM Simulator](#HELM_simulator)
- [FAME](#FAME)
  - [FAME Pipeline](#FAME_pipeline)
- [JEWEL](#JEWEL)
- [TOGA Optimization](#TOGA)

# ACME

## ACME_pipeline

For a first-time run on raw files (this will generate .pickle files for future use):

`$ ACME_pipeline --data "path/to/files/*.raw" --outdir specify/directory`

For pickle files (Scan all .pickle files in directory):

`$ ACME_pipeline --data "path/to/files/*.pickle" --outdir specify/directory`

For reprocessing the database:

`$ ACME_pipeline --reprocess_dir "labdata/toplevel/" --reprocess_version vX.y --reprocess`

### Arguments

This table lists all arguments available. They are annotated with emoji
flags to indicate the following:

- :white_check_mark: Required
- :arrow_up_small: Increased Runtime
- :arrow_down_small: Decreased Runtime
- :bulb: Useful, often used
- :exclamation: Warning, requires deliberate use

| :star: | Argument flag | Description | Default Value |
| -- | -- | -- | -- |
| :bulb: | `--data` | Glob of files to be processed | None |
| :bulb: | `--outdir` | Output directory path | None |
|    | `--reprocess` | Enables bulk processing instead of per-experiment processing | None |
|    | `--reprocess_dir` | Top level lab data directory for bulk reprocessing | (internal path) |
|    | `--reprocess_version` | Version tag for bulk reprocessing output | None |
|    | `--masses` | Path to file containing known masses | `cli/configs/compounds.yml` |
|    | `--params` | Path to config file for Analyzer | `cli/configs/acme_config.yml` |
|    | `--sue_weights` | Path to weights for Science Utility Estimate | `cli/configs/acme_sue_weights.yml` |
|    | `--dd_weights` | Path to weights for Diversity Descriptor | `cli/configs/acme_dd_weights.yml` |
|    | `--log_name` | Filename for the pipeline log | `ACME_pipeline.log` |
|    | `--log_folder` | Folder path to store logs | `cli/logs` |
| :arrow_down_small: | `--noplots` | Disables plotting output | None |
| :arrow_down_small: | `--noexcel` | Disables excel file output | None |
| :arrow_up_small: | `--debug_plots` | Enables per-peak plots for debugging purposes | None |
|    | `--skip_existing` | Skip output if output already exists | None |
| :bulb: :arrow_down_small: | `--field_mode` | Only output science products; equivalent to `--noplots --noexcel` | None |
|    | `--cores` | Number of processor cores to utilize | `7` |
|    | `--saveheatmapdata` | Save heatmap as data file in addition to image | None |
|    | `--priority_bin` | Downlink priority bin (lower number is higher priority) for generated products | `0` |
|    | `--manifest_metadata` | Manifest metadata (YAML string); takes precedence over file entries | None |
|    | `--manifest_metadata_file` | Manifest metadata file (YAML) | None |
| :bulb:  | `--knowntraces` | Process only known masses specified in `configs/compounds.yml` | None |

------

##  ACME_simulator
Simulate Raw ACME samples to debug and better understand ACME analyser.
This Simulator was used to generate the Silver and Golden Dataset at data_OWLS/ACME/....
The config file for those datasets is saved in those folders as well.

### Arguments

| Argument flag | Description | Default Value |
| -- | -- | -- |
| `--params` | Path to config file | `cli/configs/acme_sim_params.yml` |
| `--out_dir` | Path to output directory | None |
| `--n_runs` | Number of simulation runs | `10` |
| `--log_name` | Filename for pipeline log | `ACME_simulator.log` |
| `--log_folder` | Folder path to store logs | `cli/logs` |

------

## ACME_evaluation

ACME Evaluation measures the performance of ACME on simulator data and hand-labeled lab data.
There are two versions of this script:

- `ACME_evaluation.py` is the original evaluation script. It calculates the
  `output precision` and the `label recall`, which is the number of output peaks
  that match to any label and the number of labels that match to any output,
  respectively. Note that the f1-score cannot be calculated due to differing populations.
- `ACME_evaluation_script.py` is the stricter evaluation script. It enforces a
  one-to-one match between the output and labeled peaks, marking duplicate detections
  as false positives. This allows the script to calculate a formal `precision`,
  `recall`, and `f1`. **This script is recommended.**

### Arguments

| Argument flag | Description | Default Value |
| -- | -- | -- |
| `acme_outputs` | Required, glob of peak output from analyzer | None |
| `acme_labels` | Required, glob of peak labels | None |
| `--hand_labels` | Expects hand labels (as opposed to simulator labels) | None |
| `--mass_threshold` | Max distance (in mass indices) between matching peaks. | `30` |
| `--time_threshold` | Max distance (in time indices) between matching peaks. | `30` |
| `--ambiguous` | Some peaks are labeled as ambiguous. Consider these as true peaks. | None |
| `--log_name` | Filename for pipeline log. | `ACME_evaluation.log` |
| `--log_folder` | Folder path to store logs. | `cli/logs` |

------

# HELM

## HELM_pipeline

### Arguments
This table lists all arguments available. They are annotated with emoji
flags to indicate the following:

- :white_check_mark: Required
- :arrow_up_small: Increased Runtime
- :arrow_down_small: Decreased Runtime
- :bulb: Useful, often used
- :exclamation: Warning, requires deliberate use

| :star: | Argument flag | Description | Default Value |
| -- | -- | -- | -- |
| :bulb: | `--config` | Filepath of configuration file. | `cli/configs/helm_config_labtrain.yml` |
| :white_check_mark: | `--experiments` | Glob string pattern of experiment directories to process. | None |
| :white_check_mark: | `--steps` | Steps of the pipeline to run. See below for description of steps. | None |
| :white_check_mark: | `--batch_outdir` | Output directory for batch-level results. | None |
| :bulb: :arrow_down_small: | `--use_existing` | Attempt to reuse previous processing output for any steps defined here. See description below for options. | None |
| :bulb: :arrow_down_small: | `--field_mode` | Only output field products. Skips most plots. | None |
| :bulb: | `--cores` | Number of processor cores to utilize. | `7` |
|    | `--note` | String to be appended to output directory name. | None |
|    | `--log_name` | Filename for the pipeline log. | `HELM_pipeline.log` |
|    | `--log_folder` | Folder path to store logs. | `cli/logs` |
|    | `--priority_bin` | Downlink priority bin (lower number is higher priority) for generated products | `0` |
|    | `--manifest_metadata` | Manifest metadata (YAML string); takes precedence over file entries | None |
|    | `--manifest_metadata_file` | Manifest metadata file (YAML) | None |
| :exclamation: | `--train_feats` | Only usees tracks with labels for model training. | None |
| :exclamation: | `--predict_model` | Path to ML model for motility classification. | `cli/models/classifier_labtrain_v02.pickle` |
| :exclamation: | `--toga_config` | Override config filepath for TOGA optimization. | None |

### Steps

This table lists all steps available. It also indicates which steps can be used with
the `--use_existing` step. It is listed in typical order of usage.

| Step Name | Description | `--use_existing` |
| -- | -- | -- |
| `preproc` | Lowers the resolution from 2048x2048 to 1024x1024 for analysis. | TRUE |
| `validate` | Generates data validation products, including videos and MHIs. | TRUE |
| `tracker` | Track particles in the experiment. | TRUE |
| `point_evaluation` | Using track labels, measure point accuracy of the tracker. | TRUE |
| `track_evaluation` | Using track labels, measure track accuracy of the tracker. | TRUE |
| `features` | Extract features from detected tracks. | FALSE |
| `train` | Train the motility classification model. | FALSE |
| `predict` | Predict motility of all tracks with classification model. | FALSE |
| `asdp` | Generate ASDP products, including a visualization video. | FALSE |
| `manifest` | Generate file manifest for JEWEL. | FALSE |

Most steps depend on output from all previous steps. This table lists step prerequisites.

| Step Name | Prerequisite Steps | Other Reqs |
| -- | -- | -- |
| `preproc` | N/A | N/A |
| `validate` | `preproc` | N/A |
| `tracker` | `preproc` `validate` | N/A |
| `point_evaluation` | `preproc` `validate` `tracker` | Track Labels |
| `track_evaluation` | `preproc` `validate` `tracker` | Track Labels |
| `features` | `preproc` `validate` `tracker` | `track_evaluation` Optional |
| `train` | `preproc` `validate` `tracker` `track_evaluation` `features` | Track Labels |
| `predict` | `preproc` `validate` `tracker` `features` | Pretrained Model |
| `asdp` | `preproc` `validate` `tracker` `features` `predict` | Track Labels Optional |
| `manifest` | N/A | Various `validate`, `tracker`, `predict`, `asdp` Products Optional |

There are also pipelines available for common combinations of steps.

| Pipeline Name | Description | Steps |
| -- | -- | -- |
| `pipeline_train` | Pipeline to train the motility classifier. | `preproc`, `validate`, `tracker`, `track_evaluation`, `features`, `train` |
| `pipeline_predict` | Pipeline to predict motility. | `preproc`, `validate`, `tracker`, `features`, `predict` |
| `pipeline_tracker_eval` | Pipeline to evaluate tracker performance. | `preproc`, `validate`, `tracker`, `point_evaluation`, `track_evaluation` |
| `pipeline_products` | Pipeline to generate all products. | `preproc`, `validate`, `tracker`, `point_evaluation`, `track_evaluation`, `features`, `predict`, `asdp`, `manifest` |
| `pipeline_field` | Pipeline to generate field-mode products. Disables most plotting, especially in validate. | `preproc`, `validate`, `tracker`, `features`, `predict`, `asdp`, `manifest` |

### Valid Experiments

An experiment is defined by a unique directory. To be considered valid, an experiment must satisfy the following:

* Contain the subdirectory `Holograms/`
* Have at least `50` valid holograms in said subdirectory. Valid holograms are:
  * Images with extension `.tif`
  * Images with resolution `2048x2048`
  * These values can be configured in the config.
* The enumerated names of the images are expected to be consecutive.


### Common Usage Examples

A brief reminder that these examples assume you have followed the installation
instructions.

Make sure to specify desired configuration parameters in `helm_config_labtrain.yml`
before executing the pipeline. These use `src/cli/configs/helm_config_labtrain.yml`
by default.

**Validate your Experiment Data**
```bash
HELM_pipeline \
--experiments "my/experiments/glob/wildcard_*_string" \
--batch_outdir my/experiments/batch_directory \
--steps preproc validate
```
Note how, by adding a wildcard, you can process multiple experiments at once.


**Generate Particle Tracks**
```bash
HELM_pipeline \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps preproc validate tracker \
--use_existing preproc validate
```

Note how, by specifying `--use_existing`, the pipeline will use existing
`preproc` and `validate` step output if they already exist.

**Train a motility model**

Use the `pipeline_train` step bundle to run the tracker, evaluation, feature generator, and training steps. The `--use_existing` flag will skip any steps that were previously computed:

```bash
HELM_pipeline \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps pipeline_train \
--train_feats \
--use_existing preproc validate
```

**Validate, Track Particles, Predict Motility, and Generate Visualization**
```bash
HELM_pipeline \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps pipeline_products \
--use_existing preproc validate tracker
```
Note that `--config` and `--predict_model` can also be specified, but we're
just using the default values here.

### Tracker outputs
In the output folder, the following subfolders will be made:

/plots: plots of all the tracks, where each track is colored with a distinct color

/tracks: subfolders for each cases, with .track files giving the coordinates

/configs: configuration file for the case

/train classifier: an empty folder, which is needed by `train_model.py`

### Classifier outputs
In the output folder, under /train classifier, you will see the following:

track motility.csv, which gives each track with its label, is just a log file output

yourclassifier.pickle, classifier in pkl form

plots/ roc curve, confusion matrix, `feature_importance` if running Zaki's classifier

## HELM_simulator

The HELM simulator generates synthetic DHM data. It can be used as a source of
sensitivity and sanity checks to assess performance of edge cases and limits (such as
particle speed) of HELM. This tool is broken down into two major steps:
1. simulate particle tracks (specifying position, brightness, velocity through time)
2. simulate DHM images from tracks (creating 2D tif hologram images using noise and tracks)

```bash
HELM_simulator [required-args] [optional-args]
```

### Arguments

This table lists all arguments available. They are annotated with emoji
flags to indicate the following:

- :white_check_mark: Required
- :arrow_up_small: Increased Runtime
- :arrow_down_small: Decreased Runtime
- :bulb: Useful, often used
- :exclamation: Warning, requires deliberate use

| :star: | Argument flag | Description | Default Value |
| -- | -- | -- | -- |
| :bulb: | `--configs` | Configuration parameters for synthetic data. | `configs/helm_simulator_config.yml` |
| :bulb: | `--n_exp` | Number of experiments to generate with config(s). | 1 |
| :white_check_mark: | `--sim_outdir` | Directory to save the synthetic data to.  | None |
| | `--log_name` | Filename for the pipeline log. | `HELM_simulator.log` |
| | `--log_folder` | Folder path to store logs. |`cli/logs` |

### Common Usage Examples
```bash
# Single config
HELM_simulator.py \
--configs src/cli/configs/helm_simulator_config_v2.yml \
--n_exp 2 \
--sim_outdir <local_output_dir>

# Multiple configs
HELM_simulator.py \
--configs src/cli/configs/sim_config_v*.yml \
--n_exp 2 \
--sim_outdir <local_output_dir>
```

### Config options
Within the configuration file, you can set items like
* image parameters (e.g., resolution, chamber size, noise characteristics, etc.)
* experiment parameters (number of motile/non-motile particles, length of recording, drift, etc.)
* particle parameters (e.g., shape/size/brightness of particle, movement distribution, etc.)

There are two pre-baked configuration to choose from. They differ slightly in how they generate motile tracks dynamics.
1. helm_simulator_config_v1.yml
  This configuration generates all tracks (motile and non-motile) by making random perturbations to a track's velocity at each time step. Motile tracks are best generated by assigning a wide movement distribution and high momentum. This approach is simple to use, but creates tracks that are less realistic than in `helm_simulator_config_v2.yml`.

2. helm_simulator_config_v2.yml
  This configuration (default) uses Variational Autoregression (VAR) models to simulate more realistic motile tracks. Non-motile tracks are simulated using the same random perturbation approach in `helm_simulator_config_v1.yml`. VAR models are essentially N-dimensional autoregression models. Use VAR models that were fitted to real particle tracks to generate synthetic ones.

  In the config, you must specify a VAR model file that was calibrated using the `statsmodels` package. The VAR model files are stored in `helm_dhm/simulator/var_models` and an example on how to fit a VAR model to data is in `src/research/wronk/simulation_dynamics`. See [this statsmodels page](https://www.statsmodels.org/dev/vector_ar.html) for general information on VAR models.

------

# FAME

## FAME_pipeline

### Arguments
This table lists all arguments available. They are annotated with emoji
flags to indicate the following:

- :white_check_mark: Required
- :arrow_up_small: Increased Runtime
- :arrow_down_small: Decreased Runtime
- :bulb: Useful, often used
- :exclamation: Warning, requires deliberate use

| :star: | Argument flag | Description | Default Value |
| -- | -- | -- | -- |
| :bulb: | `--config` | Filepath of configuration file. | `cli/configs/fame_config.yml` |
| :white_check_mark: | `--experiments` | Glob string pattern of experiment directories to process. | None |
| :white_check_mark: | `--steps` | Steps of the pipeline to run. See below for description of steps. | None |
| :white_check_mark: | `--batch_outdir` | Output directory for batch-level results. | None |
| :bulb: :arrow_down_small: | `--use_existing` | Attempt to reuse previous processing output for any steps defined here. See description below for options. | None |
| :bulb: :arrow_down_small: | `--field_mode` | Only output field products. Skips most plots. | None |
| :bulb: | `--cores` | Number of processor cores to utilize. | `7` |
|    | `--note` | String to be appended to output directory name. | None |
|    | `--log_name` | Filename for the pipeline log. | `FAME_pipeline.log` |
|    | `--log_folder` | Folder path to store logs. | `cli/logs` |
|    | `--priority_bin` | Downlink priority bin (lower number is higher priority) for generated products | `0` |
|    | `--manifest_metadata` | Manifest metadata (YAML string); takes precedence over file entries | None |
|    | `--manifest_metadata_file` | Manifest metadata file (YAML) | None |
| :exclamation: | `--train_feats` | Only usees tracks with labels for model training. | None |
| :exclamation: | `--predict_model` | Path to ML model for motility classification. | `cli/models/classifier_labtrain_v02.pickle` |
| :exclamation: | `--toga_config` | Override config filepath for TOGA optimization. | None |

### Steps

This table lists all steps available. It also indicates which steps can be used with
the `--use_existing` step. It is listed in typical order of usage.

| Step Name | Description | `--use_existing` |
| -- | -- | -- |
| `preproc` | Lowers the resolution from 2048x2048 to 1024x1024 for analysis. | TRUE |
| `validate` | Generates data validation products, including videos and MHIs. | TRUE |
| `tracker` | Track particles in the experiment. | TRUE |
| `point_evaluation` | Using track labels, measure point accuracy of the tracker. | TRUE |
| `track_evaluation` | Using track labels, measure track accuracy of the tracker. | TRUE |
| `features` | Extract features from detected tracks. | FALSE |
| `train` | Train the motility classification model. | FALSE |
| `predict` | Predict motility of all tracks with classification model. | FALSE |
| `asdp` | Generate ASDP products, including a visualization video. | FALSE |
| `manifest` | Generate file manifest for JEWEL. | FALSE |

Most steps depend on output from all previous steps. This table lists step prerequisites.

| Step Name | Prerequisite Steps | Other Reqs |
| -- | -- | -- |
| `preproc` | N/A | N/A |
| `validate` | `preproc` | N/A |
| `tracker` | `preproc` `validate` | N/A |
| `point_evaluation` | `preproc` `validate` `tracker` | Track Labels |
| `track_evaluation` | `preproc` `validate` `tracker` | Track Labels |
| `features` | `preproc` `validate` `tracker` | `track_evaluation` Optional |
| `train` | `preproc` `validate` `tracker` `track_evaluation` `features` | Track Labels |
| `predict` | `preproc` `validate` `tracker` `features` | Pretrained Model |
| `asdp` | `preproc` `validate` `tracker` `features` `predict` | Track Labels Optional |
| `manifest` | N/A | Various `validate`, `tracker`, `predict`, `asdp` Products Optional |

There are also pipelines available for common combinations of steps.

| Pipeline Name | Description | Steps |
| -- | -- | -- |
| `pipeline_train` | Pipeline to train the motility classifier. | `preproc`, `validate`, `tracker`, `track_evaluation`, `features`, `train` |
| `pipeline_predict` | Pipeline to predict motility. | `preproc`, `validate`, `tracker`, `features`, `predict` |
| `pipeline_tracker_eval` | Pipeline to evaluate tracker performance. | `preproc`, `validate`, `tracker`, `point_evaluation`, `track_evaluation` |
| `pipeline_products` | Pipeline to generate all products. | `preproc`, `validate`, `tracker`, `point_evaluation`, `track_evaluation`, `features`, `predict`, `asdp`, `manifest` |
| `pipeline_field` | Pipeline to generate field-mode products. Disables most plotting, especially in validate. | `preproc`, `validate`, `tracker`, `features`, `predict`, `asdp`, `manifest` |

### Valid Experiments

An experiment is defined by a unique directory. To be considered valid, an experiment must satisfy the following:

* Contain the subdirectory `Holograms/`
* Have at least `50` valid frames in said subdirectory. Valid frames are:
  * Images with extension `.tif`
  * Images with resolution `2048x2048`
  * These values can be configured in the config.
* The enumerated names of the images are expected to be consecutive.


### Common Usage Examples

A brief reminder that these examples assume you have followed the installation
instructions.

Make sure to specify desired configuration parameters in `fame_config.yml`
before executing the pipeline. These use `src/cli/configs/fame_config.yml`
by default.

**Validate your Experiment Data**
```bash
FAME_pipeline \
--experiments "my/experiments/glob/wildcard_*_string" \
--batch_outdir my/experiments/batch_directory \
--steps preproc validate
```
Note how, by adding a wildcard, you can process multiple experiments at once.


**Generate Particle Tracks**
```bash
FAME_pipeline \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps preproc validate tracker \
--use_existing preproc validate
```

Note how, by specifying `--use_existing`, the pipeline will use existing
`preproc` and `validate` step output if they already exist.

**Train a motility model**

Use the `pipeline_train` step bundle to run the tracker, evaluation, feature generator, and training steps. The `--use_existing` flag will skip any steps that were previously computed:

```bash
FAME_pipeline \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps pipeline_train \
--train_feats \
--use_existing preproc validate
```

**Validate, Track Particles, Predict Motility, and Generate Visualization**
```bash
FAME_pipeline \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps pipeline_products \
--use_existing preproc validate tracker
```
Note that `--config` and `--predict_model` can also be specified, but we're
just using the default values here.

### Tracker outputs
In the output folder, the following subfolders will be made:

/plots: plots of all the tracks, where each track is colored with a distinct color

/tracks: subfolders for each cases, with .track files giving the coordinates

/configs: configuration file for the case

/train classifier: an empty folder, which is needed by `train_model.py`

### Classifier outputs
In the output folder, under /train classifier, you will see the following:

track motility.csv, which gives each track with its label, is just a log file output

yourclassifier.pickle, classifier in pkl form

plots/ roc curve, confusion matrix, `feature_importance` if running Zaki's classifier


------

# JEWEL

JEWEL generates an ordering for downlinking ASDPs contained within an ASDP
Database (ASDP DB), given a user-specified configuration file.

```bash
JEWEL [required-args] [optional-args]
```

| Required | Argument Flag | Description | Default |
| -- | -- | -- | -- |
| :white_check_mark: | `dbfile` | The path to the ASDP DB file | None |
| :white_check_mark: | `outputfile` | The path to the output CSV file where the ordered data products will be written | None |
|  | `--config` | Path to a config (.yml) file. | `cli/configs/jewel_default.yml` |
|  | `--log_name` | Filename for the pipeline log. | `JEWEL.log` |
|  | `--log_folder` | Folder path to store logs. | `cli/logs` |


## Auxiliary Scripts

Below are several several scripts used to support JEWEL by managing the ASDP DB.
First is a script to update the contents of the ASDP DB. The first time the
script is invoked, an ASDP DB will be initialized and populated. During
subsequent invocations, the DB will be updated with any new ASDPs that have been
generated.

```bash
update_asdp_db [required-args] [optional-args]
```

| Required | Argument Flag | Description | Default |
| -- | -- | -- | -- |
| :white_check_mark: | `rootdirs` | A list of root directories for each of the ASDP results (e.g., for HELM or ACME); each directory should correspond to a single ASDP. | None |
| :white_check_mark: | `dbfile` | The path to where the DB file will be stored (currently CSV format) | None |
|  | `--log_name`   | Filename for the pipeline log. | `update_asdp_db.log` |
|  | `--log_folder` | Folder path to store logs. | `cli/logs` |

The next script simulates a downlink for testing JEWEL with ground in the loop.
It traverses the ordering produced by JEWEL, "downlinks" untransmitted ASDPs,
and marks them as transmitted within the ASDP DB.

```bash
simulate_downlink [required-args] [optional-args]
```

| Required | Argument | Description | Default |
| -- | -- | -- | -- |
| :white_check_mark: | `dbfile` | The path to the ASDP DB file | None |
| :white_check_mark: | `orderfile` | The path to the ASDP ordering file produced by JEWEL | None |
| :white_check_mark: | `datavolume` | The simulated downlink data volume in bytes | None |
|  | `-d`/`--downlinkdir` | Simulate downlink of ASDP files by copying to this directory. If None, still mark files as downlinked. | None |
|  | `--log_name` | Filename for the pipeline log | `simulated_downlink.log` |
|  | `--log_folder` | Folder path to store logs. | `cli/logs` |

Finally, the last script is used to manually set the downlink status of
individual ASDPs. This can be invoked during the downlink process, or via a
ground command to manually reset the downlink status of an item that was
transmitted but not received, for example.

```bash
set_downlink_status [required-args] [optional-args]
```

| Required | Argument | Description | Default |
| -- | -- | -- | -- |
| :white_check_mark: | `dbfile` | The path to the ASDP DB file. | None |
| :white_check_mark: | `asdpid` | Integer ASDP identifier. | None |
| :white_check_mark: | `status` | The new downlink status, either "transmitted" or "untransmitted". | None |
|  | `--log_name` | Filename for the pipeline log. | `set_downlink_status.log` |
|  | `--log_folder` | Folder path to store logs. | `cli/logs` |

# TOGA

## Running TOGA on HELM

Generic docs for getting toga installed here: https://github-fn.jpl.nasa.gov/MLIA/TOGA/blob/master/README.md

```bash
git clone https://github-fn.jpl.nasa.gov/MLIA/TOGA.git
cd TOGA
conda env create -f envs/toga36-env.yml
source activate toga36
python setup.py develop
```

`python setup.py develop` is critical here, not `pip install .` . This will set up the package such that the installation uses code from this directory instead of copying files to the pip library.

`cli/TOGA_wrapper.py` is the main interface between HELM and TOGA.
This script reads in a TOGA generated config file along with the experiment
directory to run on. It then calls HELM_pipeline via subprocess and reports
back to TOGA via a generated `metrics.csv` file.

In addition to usual TOGA parameters a subset of point and/or track evaluation metrics, `metrics_names`, must be specified
in the config (as a list). These metrics are each aggregated over the experiments via a simple mean. In the case of multiple metrics, TOGA will treat all but one as "fixed axes" - toga does not optimize over fixed axes, rather it searches for top solutions according to the single "non-fixed" axis across the full spectrum of the others. See the banana-sink example on the toga side for a simple multi-dimensional problem.

## Steps on the TOGA side

Since we are committed to maintaining TOGA as a project agnostic tool, a few
configuration tweaks are needed specific to running TOGA on HELM. These are
mostly handled via TOGA configuration files, discussed below.

After cloning TOGA, all HELM configuration files are in
`TOGA/test/run_configurations/HELM/`. Each of `run_settings.yml`,
`gene_performance_metrics.yml`, and `genetic_algorithm_settings.yml`
should be copied to `TOGA/toga/config/` to override default TOGA
configuration.

Furthermore, the following items in `run_settings.yml` need to be updated:

- `gene_template` should be the absolute path to
`TOGA/test/run_configurations/HELM/helm_config.yml` (the only config not
copied to the `TOGA/toga/config/` folder)
- `work_dir -> base_dir` should name a working directory
- `command -> cmd` should have the absolute path to `TOGA_wrapper.py`
- `command -> static_args` should name a valid experiment dir

~IMPORTANT: These configs will then need to be copied to the toga environment lib directory to take effect. (See https://github-fn.jpl.nasa.gov/MLIA/TOGA/issues/5) for details.~ Issue closed.

Lastly, the environment variable `PYTHON` needs to be set (in the TOGA
virtual environment) to the python executable running HELM. This is the
version of python `TOGA_wrapper.py` will use to run the helm pipeline.

```bash
which python
export PYTHON="absolute/path/to/conda/python"
```

TODO: Helm should have its own virtual envrionment to avoid the environment
variable.

## Thomas's debugging tips

* Important: For MLIA machines, you may need to copy config files to wherever TOGA is installed upon update before calling toga_server or toga_client. For code changes (inserting prints/etc.), rerun pip install on TOGA. It is easy to get into a state where your work repo does not match the install version that commands `toga_server`/`toga_client` reference.

* On starting TOGA client, does the helm pipeline run? I.e. does the client start spewing helm prints?
  * Yes? Continue below
  * No? Look in toga's experiment directory (`work_dir` -> `base_dir` specified in run_settings). Are there any yml's in the `random_config` subdir?
    * Yes? TOGA should be able to call HELM. Put prints in TOGA_wrapper.py on the helm side.
    * No? TOGA is failing to generate configs. Ensure the helm config yml specifies genes correctly (properly indented, type and range for each). Put prints in population.py -> `create_individual()` and `mutate()`. TODO: "bool" type is broken; use int in range [0, 1] instead. Issue logged on TOGA side
* Try running with a single worker on a single experiment (so runtime is short). Upon finishing the helm script does the client print "{'individual': [uuid], 'status': 'successfully stored'}"?
  * Yes? The client seems to be running correctly. If configs do not appear in the `best` toga experiment subdir, the server may not yet have updated (it prints "running serialization" when this happens) or it may need to be restarted for any config changes to take affect.
  * No? If it looks like the HELM run was cut short, the timeout in run_settings.yml may be set too low. Otherwise, the client is probably failing to parse the outputted metrics after finishing the call to HELM. Prints can be placed at the bottom of TOGA_wrapper.py. Double check that the `metric_names` key in helm_config.yml matches those in gene_performance_metrics.yml.

## Jake's debugging tips
* Any warnings that the shell is not configured for conda can be safely ignored.
* The TOGA client will likely not quit on `Ctrl-C`, use screen then use `Ctrl-A`, `K`, then `Y` to terminate.
* Don't tell TOGA to use conda in the run settings on `analysis` or `paralysis`
