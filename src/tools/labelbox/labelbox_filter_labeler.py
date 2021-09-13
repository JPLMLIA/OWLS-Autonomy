'''
Script to only retrieve a single labeler's labels from an exported JSON
'''
import os
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('exported_json',    help='Filepath of exported labelbox JSON')

    parser.add_argument('username',     help='Username of labeler to be filtered for. Only supports a single labeler.')

    parser.add_argument('--outfile',        help='Filepath of filtered JSON to be written',
                                            default='filtered.json')
    
    args = parser.parse_args()

    with open(args.exported_json, 'r') as f:
        rows = json.load(f)

    filtered = [] 
    for row in rows:
        row_username = row['Created By'].split('@')[0]
        if row_username == args.username:
            filtered.append(row)

    with open(args.outfile, 'w') as f:
        json.dump(filtered, f)