'''
Generate data package for use in LabelBox
'''
import json
import sys
import os
import argparse
import shutil
import yaml

from glob import glob

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name',           required=True,
                                                    help="Name to use in LabelBox for this set of data.")

    parser.add_argument('--experiment_glob',        required=True,
                                                    help="Glob of experiments to include in this LabelBox data delivery.")

    parser.add_argument('--labelbox_staging_dir',   required=True,
                                                    help="Empty directory to stage data to be uploaded to LabelBox")

    parser.add_argument('--config',                 required=True,
                                                    help="Config used to generate source data (used to get subdir names)")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    json_list = []

    experiments = list(glob(args.experiment_glob))
    for experiment in experiments:

        experiment_name = experiment.split("/")[-1]
        experiment_name_nodot = experiment_name.replace(".", "_")
        experiment_labelbox_path = os.path.join(args.labelbox_staging_dir, experiment_name_nodot)
        
        movie_in = os.path.join(experiment, 
                                config['experiment_dirs']['validate_dir'], 
                                f"{experiment_name}_base_movie.mp4")

        movie_out = os.path.join(experiment_labelbox_path, 
                                 f"{experiment_name_nodot}_base_movie.mp4")

        mhi_in = os.path.join(experiment, 
                              config['experiment_dirs']['validate_dir'], 
                              f"{experiment_name}_mhi_labeled.png")

        mhi_out = os.path.join(experiment_labelbox_path, 
                               f"{experiment_name_nodot}_mhi_labeled.png")

        if not os.path.exists(experiment_labelbox_path):
            os.makedirs(experiment_labelbox_path)

        shutil.copy(movie_in, movie_out)
        shutil.copy(mhi_in, mhi_out)

        json_segment = {
         "externalId": experiment_name_nodot,
         "videoUrl": f"https://ml.jpl.nasa.gov/owls_labeling/{experiment_name_nodot}/{experiment_name_nodot}_base_movie.mp4",
         "attachments": [
             {
                 "type": "IMAGE",
                 "value": f"https://ml.jpl.nasa.gov/owls_labeling/{experiment_name_nodot}/{experiment_name_nodot}_mhi_labeled.png"
             }
         ]
        }

        json_list.append(json_segment)

    with open(os.path.join(args.labelbox_staging_dir, args.dataset_name + ".json"), 'w') as outfile:
        json.dump(json_list, outfile)








