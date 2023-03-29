## Classifiers

To train your own classifier, see examples in
`src/experiments/wronk/classifier_training/ml_motility_classifier_training_v2.ipynb`

Notably, make sure to:
* embed any preprocessing (e.g., standard scaler for SVCs) into a sklearn pipeline object
* save your model/pipeline as a pickle file
* save the feature column names along with your model like `(classifier, train_feat_names)`

### classifier_labtrain_v02.pickle
Trained on all data from `/data/MLIA_active_data/data_OWLS/HELM/data/lab/train/`
using new set of feature calculations from improvements made in April 2021.

### classifier_labelbox_RF_v01.pickle
Trained on all data from `/data/MLIA_active_data/data_OWLS/HELM/data/lab/labeled`
Using new set of feature calculations from improvements as of August 2021.

### classifier_labelbox_RF_v02.pickle
Trained on all DHM data from `/data/MLIA_active_data/data_OWLS/HELM/data/lab/labeled`
Using new set of feature calculations from improvements as of August 2021 and using
a Bayesian HP optimization scheme in
`src/experiments/wronk/classifier_training/ml_motility_classifier_training_v2.ipynb`.
Also see performance plots in that same directory to compare against other classifiers.
Trained in April, 2022.

### classifier_labelbox_GBT_v02.pickle
Trained on all data from `/data/MLIA_active_data/data_OWLS/HELM/data/lab/labeled`
Using new set of feature calculations from improvements as of August 2021 and using
a Bayesian HP optimization scheme in
`src/experiments/wronk/classifier_training/ml_motility_classifier_training_v2.ipynb`.
Also see performance plots in that same directory to compare against other classifiers.
Trained in April, 2022.

### classifier_labelbox_SVC_v02.pickle
Trained on all data from `/data/MLIA_active_data/data_OWLS/HELM/data/lab/labeled`
Using new set of feature calculations from improvements as of August 2021 and using
a Bayesian HP optimization scheme in
`src/experiments/wronk/classifier_training/ml_motility_classifier_training_v2.ipynb`.
Also see performance plots in that same directory to compare against other classifiers.
Trained in April, 2022.
