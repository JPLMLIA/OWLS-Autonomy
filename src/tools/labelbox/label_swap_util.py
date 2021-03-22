'''
Utility script to swap old label format to LabelBox format
'''
import os
import argparse

import pandas as pd
import numpy  as np

from glob  import glob
from numpy import genfromtxt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiments_glob',  required=True,
                                               help='Glob of experiments which need labels swapped')

    parser.add_argument('--label_subdir',      required=True,
                                               help='Subdir within each experiment for which labels exist')

    args = parser.parse_args()

    experiments = list(glob(args.experiments_glob))
    for experiment in experiments:

        experiment_name = experiment.split("/")[-1]
        label_path = os.path.join(experiment, args.label_subdir)
        label_in_path = os.path.join(label_path, f"verbose_{experiment_name}.csv")
        old_data = pd.read_csv(label_in_path, sep=',').values
        samples, old_features = old_data.shape

        points = []
        for x in range(0, samples):
            point = {"frame":old_data[x,3], "track":old_data[x,0], "X":old_data[x,1], "Y":old_data[x,2], "motility":old_data[x,5].lower()}
            points.append(point)

        df = pd.DataFrame(points)
        df.to_csv(os.path.join(label_path,f"{experiment_name}_labels.csv"), index=False, header="frame,track,X,Y,motility")