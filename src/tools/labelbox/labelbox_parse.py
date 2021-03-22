'''
Parse labels from LabelBox for use in OWLS pipeline
'''
import requests
import json
import sys
import os
import argparse
import shutil
import yaml

import numpy  as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--api_key',                required=True,
                                                    help="LabelBox API key")

    parser.add_argument('--label_metadata_file',    required=True,
                                                    help="label metadata file downloaded from LabelBox")

    parser.add_argument('--experiment_dir',         required=True,
                                                    help="Directory to place per-experiment label files")

    args = parser.parse_args()

    label_metadata = json.load(open(args.label_metadata_file, 'rb'))
    num_experiments = len(label_metadata)

    for x in range(0, num_experiments):
        filename = label_metadata[x]['External ID'] + "_labels.csv"
        frames_url = label_metadata[x]["Label"]['frames']

        headers = {'Authorization': f"Bearer {args.api_key}"}
        r = requests.get(frames_url, headers=headers)

        data = r.text.split('\n')

        points = []
        trackNumberMap = {}
        trackCount = 0
        for frame in data:
            
            try:
                frame = json.loads(frame)
                trackPoints = frame["objects"]
                frameNumber = frame['frameNumber']
                for trackPoint in trackPoints:
                    if trackPoint['featureId'] in trackNumberMap.keys():
                        trackNumber = trackNumberMap[trackPoint['featureId']]
                    else:
                        trackNumber = trackCount
                        trackNumberMap[trackPoint['featureId']] = trackCount
                        trackCount += 1

                    point = {"frame":frameNumber, "track":trackNumber, "X":trackPoint["point"]["x"], "Y":trackPoint["point"]["y"], "motility":trackPoint['value']}
                    points.append(point)
            except:
                pass

        df = pd.DataFrame(points)
        df.to_csv(os.path.join(args.experiment_dir, filename), index=False, header="frame,track,X,Y,motility")








