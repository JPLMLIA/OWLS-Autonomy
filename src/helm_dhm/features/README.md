# Features Extraction
The `features` step in HELM_pipeline calculates feature metrics that quantitatively characterize a particle's track. HELM will save out a CSV file (at `<experiment_directory>/features`) containing the value for each feature metric for all tracks after the `features` step completes. These features are used by the downstream machine learning algorithm (in the `predict` step) to classify if the tracked particle appears motile or not.

# Feature Descriptions
The full technical description and equations for each feature metric are in the [features jupyter notebook](feature_descriptions.ipynb).