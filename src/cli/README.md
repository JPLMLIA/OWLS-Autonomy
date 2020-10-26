# Instructions for running the code

The following code is to be run from the cli directory.
```
cd cli/
```

## Running ACME pipeline

For raw files (Scan all .raw files in directory):

	python ACME_pipeline.py --data "*\*\*.raw" --outdir <specify directory>

For pickle files (Scan all .pickle files in directory):

	python ACME_pipeline.py --data "*\*\*.pickle" --outdir <specify directory>

------
### Additional running options:

    --reprocess_dir         Top level lab data directory for bulk reprocessing
    
    --reprocess_version     Version tag for bulk reprocessing.
    
    --knowntraces           Process only known masses specified in configs/compounds.yml')

### Internal files
    --masses                Path to file containing known masses
    
    --params                Path to config file for Analyser
    
    --sue_weights           Path to weights for Science Utility Estimate')
    
    --dd_weights            Path to weights for Diversity Descriptor')

### Flags
    --noplots               If True, does not save resulting plots
    
    --noexcel               If True, does not save final excel file containing analysis information
    
    --debug_plots           If True, generates extra plots for debugging purposes
    
    --reprocess             Bulk processing of the lab data store inside data_OWLS
    
    --field_mode            Only outputs science products. (Fast runtime!)
    
    --cores                 How many processor cores to utilize

------

## Running ACME simulator
Simulate Raw ACME samples to debug and better understand ACME analyser. 
This Simulator was used to generate the Silver and Golden Dataset at data_OWLS/ACME/.... 
The config file for those datasets is saved in those folders as well

### Running parameters
    --params                Path to config file for Simulator

    --out_dir               Path to save output of Simulator

    --n_runs                Number of simulation runs to perform

------

## Running ACME evaluation
Evaluates the performance of ACME analyser on data from ACME_simulator and hand labels created from lab data
Calculates Precision, Recall, and F1 score

### Running parameters
    --analyser_outputs      Found peaks from analyzer -- Passed as globs
    
    --path_labels           Labels to compare found peaks to -- Passed as globs
    
    --hand_labels           Expects hand labels in --path_labels
    
    --zscore                Will evaluate performance for different z-score thresholds. Currently not supported for hand labels
    
    --mass_threshold        How far can peaks be apart from each other in mass [mass index] to be considered the same peak
                            '12 mass index correspond to 1 amu
    --time_threshold        How far can peaks be apart from each other in time [time index] to be considered the same peak
                            '164 time index correspond to 1 Min
    
    
------

## Running HELM pipeline
```
python HELM_pipeline.py [required-args][optional-args]
```
**`required-args`**:

`--steps` - The pipeline steps to run. A subset of `validate`, `tracker`,
`point_evaluation`, `features`, `train`, `predict`, `asdp`, or `all` to run all
steps. You can also use step bundles `pipeline_train`, `pipeline_predict`,
`pipeline_tracker_eval`, or `pipeline_products` to run common sets of steps.

`--experiments` - A list of glob-able strings to match any number of experiment
directories to include in this run. Wrap wildcard paths in quotation marks.

`--batch_outdir` - Location to write batch statistics for the run. A new timestamped dir will be created within this dir.

**`optional-args`**:

`--config` - Path to a config (.yml) file. The default config is `cli/configs/helm_config_labtrain.yml`

`--cores` - The number of cores to use.

`--use_existing` - Allow the pipeline to use previous step output if possible. Available for the `validate`, `tracker`, `point_evaluation`, `track_evaluation`

`--note` - Appends a note to the timestamped dir created within `batch_outdir`, after an underscore.

`--from_tracks` - `features` step extracts features from `tracker` output instead of existing labels. Recommended for the `train` step, but useful for debugging without the `tracker` step.

`--train_feats` - `features` step only extracts features from `tracker` output that matched with existing labels during `track_evaluation`. Required for the `train` step.

`--predict_model` - Absolute path to the pretrained model `.pickle` to be used for the `predict` step. Required for the `predict` step. Example models are provided in `models/`.

### Valid Experiments

An experiment is defined by a unique directory. To be considered valid, an experiment must satisfy the following:

* Contain a holograms subdirectory with name specified by `hologram_dir` in the config
* Have at least `min_holograms` (specified in the config) valid holograms in said holograms directory. Valid holograms must:
    * Have an extension matching one of config's `hologram_file_extensions`
    * Have an image resolution matching config's `raw_hologram_resolution`
* Have at least `min_distinct_holograms` distinct hologram images


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

### Examples
For these, make sure to specify desired configuration parameters in
`helm_config.yml` before executing the pipeline. The below examples assume your
terminal's current working directory is the `OWLS-Autonomy` repo.

Run validation step to sanity check experiment data:
```bash
python src/cli/HELM_pipeline.py \
--config src/cli/configs/helm_config_labtrain.yml \
--experiments "my/experiments/glob/wildcard_*_string" \
--batch_outdir my/experiments/batch_directory \
--steps validate
```

Run data validation and generate particle tracks:
```bash
python src/cli/HELM_pipeline.py \
--config src/cli/configs/helm_config_labtrain.yml \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps validate tracker
```

Use the `pipeline_train` step bundle to run the tracker, evaluation, feature
generator, and training. The `use_existing` flag will skip any steps that were
previously computed:
```bash
python src/cli/HELM_pipeline.py \
--config src/cli/configs/helm_config_labtrain.yml \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps pipeline_train \
--train_feats \
--from_tracks \
--use_existing
```

Run motility prediction on new data with the `pipeline_predict` step bundle:
```bash
python src/cli/HELM_pipeline.py \
--config src/cli/configs/helm_config_labtrain.yml \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps pipeline_predict \
--train_feats \
--from_tracks \
--predict_model models/classifier_labtrain.pickle
```

Run motility prediction on data with existing tracker output:
```bash
python src/cli/HELM_pipeline.py \
--config src/cli/configs/helm_config_labtrain.yml \
--experiments my/experiments/glob/string \
--batch_outdir my/experiments/batch_directory \
--steps pipeline_predict \
--train_feats \
--from_tracks \
--use_existing tracker \
--predict_model models/classifier_labtrain.pickle
```

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

### Steps on the TOGA side

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

#### Thomas's debugging tips

* Important: For MLIA machines, you may need to copy config files to wherever TOGA is installed upon update before calling toga_server or toga_client. For code changes (inserting prints/etc.), rerun pip install on TOGA. It is easy to get into a state where your work repo does not match the install version that commands `toga_server`/`toga_client` reference.

* On starting TOGA client, does the helm pipeline run? I.e. does the client start spewing helm prints?
  * Yes? Continue below
  * No? Look in toga's experiment directory (`work_dir` -> `base_dir` specified in run_settings). Are there any yml's in the `random_config` subdir?
    * Yes? TOGA should be able to call HELM. Put prints in TOGA_wrapper.py on the helm side. 
    * No? TOGA is failing to generate configs. Ensure the helm config yml specifies genes correctly (properly indented, type and range for each). Put prints in population.py -> `create_individual()` and `mutate()`. TODO: "bool" type is broken; use int in range [0, 1] instead. Issue logged on TOGA side
* Try running with a single worker on a single experiment (so runtime is short). Upon finishing the helm script does the client print "{'individual': [uuid], 'status': 'successfully stored'}"?
  * Yes? The client seems to be running correctly. If configs do not appear in the `best` toga experiment subdir, the server may not yet have updated (it prints "running serialization" when this happens) or it may need to be restarted for any config changes to take affect.
  * No? If it looks like the HELM run was cut short, the timeout in run_settings.yml may be set too low. Otherwise, the client is probably failing to parse the outputted metrics after finishing the call to HELM. Prints can be placed at the bottom of TOGA_wrapper.py. Double check that the `metric_names` key in helm_config.yml matches those in gene_performance_metrics.yml.

#### Jake's debugging tips
* Any warnings that the shell is not configured for conda can be safely ignored.
* The TOGA client will likely not quit on `Ctrl-C`, use screen then use `Ctrl-A`, `K`, then `Y` to terminate.
* Don't tell TOGA to use conda in the run settings on `analysis` or `paralysis`
