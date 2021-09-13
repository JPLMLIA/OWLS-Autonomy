"""
Upload a dataset to S3

This script will first-time upload a dataset to the S3 bucket.

You MUST have AWS CLI installed and credentialed, likely on your local machine.

Standard AWS user tokens expire after 4 hours. Run:
$ aws-login.darwin.amd64 --region us-west-1
to refresh your token.

AWS CLI Installation:
https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html

AWS CLI Credential Setup:
https://github.jpl.nasa.gov/cloud/Access-Key-Generation/blob/master/README.md

Boto3 Documentation:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
"""
import sys
import os
import os.path as op
import argparse
import logging
from glob import glob
from pathlib import Path
from tqdm import tqdm

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Modified from Boto3 documentation
def upload_file(file_name, bucket, session, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = session.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_dir',              help="Directory of experiments to upload to S3")
    parser.add_argument('--bucket',                 default="owls-autonomy",
                                                    help="Name of AWS S3 bucket to upload data to")
    parser.add_argument('--aws_profile',            default="saml-pub",
                                                    help="Name of AWS profile to use for credentials")

    args = parser.parse_args()

    if not op.isdir(args.dataset_dir):
        logging.error(f"{args.dataset_dir} is not a directory.")
        sys.exit(1)

    dataset_name = Path(args.dataset_dir).name
    
    exp_dirs = glob(op.join(args.dataset_dir, "*", ""))
    
    # Set up S3 session
    session = boto3.Session(profile_name=args.aws_profile)

    # Iterate through experiments
    S3_manifest = []
    for exp_path in exp_dirs:
        exp_id = Path(exp_path).name

        # Video File
        vid_path = op.join(exp_path, f"{exp_id}_movie.mp4")

        if not op.exists(vid_path):
            logging.error(f"{Path(vid_path).name} doesn't exist, aborting.")
            sys.exit(1)

        S3_manifest.append({
            "file_name": vid_path,
            "object_name": f"{dataset_name}/{exp_id}_movie.mp4"
        })

        # MHI File 
        mhi_path = op.join(exp_path, f"{exp_id}_mhi.png")

        if not op.exists(mhi_path):
            print(f"{Path(mhi_path).name} doesn't exist, aborting.")
            sys.exit(1)

        S3_manifest.append({
            "file_name": mhi_path,
            "object_name": f"{dataset_name}/{exp_id}_mhi.png"
        })
    
    # Upload files to S3
    for S3_obj in tqdm(S3_manifest, desc=f"Uploading to {args.bucket}"):
        upload_file(S3_obj['file_name'], args.bucket, session, S3_obj['object_name'])