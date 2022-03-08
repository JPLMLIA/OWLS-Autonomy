# How to tune Hyperparameters of ACME  
The main program to tune hyperparameters is 'dev_tune_hyperparam.py'. This program iteratively updates the acme_config.yml with randomly chosen values. 
Then the acme pipeline is run and the found peaks are compared to labels. Then a plot is generated that shows the performance of each run (precision, recall, and F1 score). The saved config file allows to investigate the used parameters.
After a series of runs (e.g. 10) one can choose the config file that gave the best performance as the new default acme config file (this is a manual process).

### Good To Know
It also helps to compare multiple runs that have similar high performance and extract (by comparison) which parameters seem to be fixed and which parameters seem to be free.
The user can then decide to run more experiments where only the free parameters are randomly sampled.

### Usage
There are few things required to run this script
 - A config file (e.g. a copy of cli/configs/acme_config.yml)
 - Pickle files of samples we want to optimize our performance on
 - Labels for those same files
 
 Once these three items are available and their path is specified in the script the user can decide which variables in the config file he wishes to tune.
 This is done by commenting out variables that should not be altered.
 
 ps.: the script is not actively maintained and not part of the pipeline, thus, some components might need some updates for their file location etc.
 
 ### Questions?
 steffen.mauceri@jpl.nasa.gov