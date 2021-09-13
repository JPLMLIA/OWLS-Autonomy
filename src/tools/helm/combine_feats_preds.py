import os
import os.path as op
from glob import glob
import argparse

import csv
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('feat_csv',         help='Filepath to features CSV of experiment.')
    parser.add_argument('pred_dir',         help='Path to predict directory of experiment.')
    parser.add_argument('--out_csv',        default='combined.csv',
                                            help='Filepath to output CSV with class column. Defaults to combined.csv')

    args = parser.parse_args()

    # Open CSV file
    with open(args.feat_csv, 'r') as f:
        reader = csv.DictReader(f)
    
        # Build new CSV file with added last column w/ motility
        output = []
        for row in reader:
            newrow = row.copy()
            track_id = newrow['track']
            pred_json_fp = op.join(args.pred_dir, f"{int(track_id):05d}.json")
            with open(pred_json_fp, 'r') as f:
                track_data = json.load(f)
            newrow['pred_motile'] = track_data['classification'] == "motile"
            output.append(newrow)
    
    # Write to output CSV
    with open(args.out_csv, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader()
        for row in output:
            writer.writerow(row)

