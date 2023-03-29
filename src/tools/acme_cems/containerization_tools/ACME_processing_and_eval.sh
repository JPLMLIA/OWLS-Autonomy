#!/bin/bash

# See the README file in this dir or in src/cli for docs for parameters that will be passed to
# ACME_flight_pipeline and ACME_evaluation_strict

echo "ACME Processing Parameters"
echo "--------------------------"
echo "data: $1"
echo "outdir: $2"
echo "params: $3"
echo "acme_outputs: $4"
echo "acme_labels: $5"
echo "log_folder: $6"

# Run ACME's analysis on the labeled dataset
echo "Beginning ACME_flight_pipeline call..."
ACME_flight_pipeline --data "$1" --outdir $2 --params $3 --log_folder $6

# Run evaluation
# Takes acme_outputs and acme_labels as input arguments
echo "Beginning ACME_evaluation_strict call..."
ACME_evaluation_strict "$4" "$5" --hand_labels --log_folder $6

echo "Processing Complete."