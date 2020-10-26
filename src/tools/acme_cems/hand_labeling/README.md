Everything you need to generate hand labels for ACME

1. Generate plots from ACME data with 'plots_ACME_for_hand_labeling.py'

2. Label Plots in MATLAB 'image labeler' (draw a box around each peak)

3. Export labels with 'convert_labels_to_csv.m'

4. Add z-score to labels peaks with 'hand_label_post_processing.py'

Enjoy your new hand labels and check your performance with 'src/cli/ACME_evaluation.py'