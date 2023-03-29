# Parameters
The data, outdir, and params parameters are passed to ACME_flight_pipeline.
The acme_outputs, acme_labels, and log_folder parameters are passed to ACME_evaluation_strict.

1. data: Path or wildcard to pickle files containing labeled data
1. outdir: Directory to save peak detections to
1. params: Path to ACME configuration file
1. acme_outputs: Path to prediction files (which will be stored in `outdir`)
1. acme_labels: Path or wildcard to hand-generated peak labels.
1. log_folder: Directory to store logs.

# Using Docker to run ACME processing

To process and evaluate a set of experiments, use something like:
```
# Build docker from the Dockerfile at the root of this repository
docker build . -f Dockerfile_ACME --tag owls-autonomy-acme:v1

# Run the container (including a volume mount so data is accessible)
docker run -it --rm \
--volume /Users/wronk/Data/OWLS-Autonomy:/Users/wronk/Data/OWLS-Autonomy \
--volume /Users/wronk/Builds/OWLS-Autonomy:/Users/wronk/Builds/OWLS-Autonomy \
owls-autonomy-acme:v1 \
"/Users/wronk/Data/OWLS-Autonomy/ACME/Hand_Labels_2021/pickles/round_1/*.pickle" \
"/Users/wronk/Data/OWLS-Autonomy/ACME/Hand_Labels_2021/ACME_preds/" \
"/Users/wronk/Builds/OWLS-Autonomy/src/cli/configs/acme_config.yml" \
"/Users/wronk/Data/OWLS-Autonomy/ACME/Hand_Labels_2021/ACME_preds/19*/*_peaks.csv" \
"/Users/wronk/Data/OWLS-Autonomy/ACME/Hand_Labels_2021/labels/round_1/*.csv" \
"/Users/wronk/Data/OWLS-Autonomy/ACME/Hand_Labels_2021/ACME_eval_logs_master/"
```

# Using Singularity to Run ACME processing and eval
```
# Save out the docker image and send it to the target machine for conversion
docker save -o owls-autonomy-acme_v2.tar owls-autonomy-acme
scp owls-autonomy-acme_v2.tar wronk@gattaca:/scratch_sm/owls-dev/wronk/acme_docker_image/owls-autonomy-acme_v2.tar

# Build the image on the target machine
module load singularity-gattaca/compute
singularity build owls-autonomy-acme-v2.sif docker-archive:///scratch_sm/owls-dev/wronk/acme_docker_image/owls-autonomy-acme_v2.tar

# Load singularity (if needed) and create an env var to set the volume mount
module load singularity-gattaca/compute
export SINGULARITY_BIND="/scratch_sm/owls-dev/wronk/Data/Hand_Labels_2021"

# Execute the singularity container
singularity exec --workdir /app \
/scratch_sm/owls-dev/wronk/acme_docker_image/owls-autonomy-acme-v2.sif \
/app/ACME_processing_and_eval.sh \
"/scratch_sm/owls-dev/wronk/Data/Hand_Labels_2021/pickles/round_1/*.pickle" \
"/scratch_sm/owls-dev/wronk/Data/Hand_Labels_2021/ACME_preds/" \
"/home/wronk/Builds/OWLS-Autonomy/src/cli/configs/acme_config.yml" \
"/scratch_sm/owls-dev/wronk/Data/Hand_Labels_2021/ACME_preds/19*/*_peaks.csv" \
"/scratch_sm/owls-dev/wronk/Data/Hand_Labels_2021/labels/round_1/*.csv" \
"/scratch_sm/owls-dev/wronk/Data/Hand_Labels_2021/ACME_eval_logs_master/"
```