"""
Upload or update a Labelbox dataset with S3 signed urls.

You MUST have AWS CLI installed and credentialed, likely on your local machine.
You MUST have a READ-ONLY service account set up on your machine that is allowed
to sign URLs for 7 days. 
You MUST have a Labelbox API key generated, and the Labelbox python SDK installed.

Contact Jake Lee for the AWS service account credentials.

AWS CLI Installation:
https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html

Adding a named profile:
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html

Boto3 Documentation:
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html

Labelbox Python SDK Documentation:
https://labelbox-python.readthedocs.io/en/latest/
https://docs.labelbox.com/python-sdk/en/index-en

Labelbox GraphQL API Documentation:
https://docs.labelbox.com/graphql-api/en/index-en
"""
import sys
import os
import os.path      as op
from pathlib        import Path
import argparse
import yaml
from tqdm           import tqdm
import re
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import boto3
from botocore.config import Config
from labelbox       import Client, Dataset

def get_s3_keys(s3_client, bucket, prefix=""):
    """ Get list of objects in specified S3 bucket
    """
    s3_response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix
    )
    s3_object_keys = [o['Key'] for o in s3_response['Contents']]

    return s3_object_keys

def get_s3_url(s3_client, bucket, key, expires_in=604800):
    """ Generate signed URL for specified object key in bucket
    """

    signed_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': bucket,
                        'Key': key
                    },
                    ExpiresIn=expires_in
    )

    return signed_url

def update_lb_attachment_url(lb_client, lb_datarow, new_attachurl):
    """ Edit the attachment of a row using direct GraphQL calls 

    This function uses client.execute() to make GraphQL calls. While
    undocumented, Labelbox support confirmed that it's a valid method of
    performing operations not documented in the SDK.
    """

    row_id = lb_datarow.uid

    # Get ID of associated attachment with GraphQL query
    attach_response = lb_client.execute("""
        query GetAttachmentByDataRowId($row_id: ID!) {
            dataRow (where: {id: $row_id}) {
                id
                attachments {
                    id
                }
            }
        }
    """, {"row_id": row_id})
    attach_id = attach_response['dataRow']['attachments'][0]['id']

    # Update URL of associated attachment with GraphQL mutation
    lb_client.execute("""
        mutation UpdateAttachment($attach_id: ID!, $attach_url: String!) {
            updateDataRowAttachment(
                where: {id: $attach_id},
                data: {
                    value: $attach_url,
                    type: IMAGE
                }
            ) {
                id
            }
        }
    """, {"attach_id": attach_id, "attach_url": new_attachurl})

    return

def upload_dataset(lb_client, s3_client, config):

    # Get list of objects in specified S3 bucket
    s3_object_keys = get_s3_keys(s3_client, config['s3_bucket'], config['s3_prefix'])

    # Sort these objects into experiments and movie/mhi pairs
    s3_exp_obj_store = {}
    for k in s3_object_keys:
        curr_exp_id = re.match(r'(.*)_', Path(k).stem)[1]
        if curr_exp_id not in s3_exp_obj_store:
            s3_exp_obj_store[curr_exp_id] = {}

        if 'movie' in k:
            s3_exp_obj_store[curr_exp_id]['movie'] = k
        elif 'mhi' in k:
            s3_exp_obj_store[curr_exp_id]['mhi'] = k
        else:
            logging.error(f"Unexpected key {k}")
            sys.exit(1)

    # Create new dataset on Labelbox
    new_dataset = lb_client.create_dataset(name=config['lb_dataset'])

    # Construct new datarows with signed URLs and attachment
    for exp_id in tqdm(s3_exp_obj_store, desc="Adding rows"):
        
        movie_signed_url = get_s3_url(s3_client, config['s3_bucket'], 
                                        s3_exp_obj_store[exp_id]['movie'],
                                        expires_in=604800)
        mhi_signed_url = get_s3_url(s3_client, config['s3_bucket'],
                                        s3_exp_obj_store[exp_id]['mhi'],
                                        expires_in=604800)

        new_dataset.create_data_row(row_data=movie_signed_url, external_id=exp_id)
        new_row = new_dataset.data_row_for_external_id(exp_id)
        new_row.create_attachment("IMAGE", mhi_signed_url)

    return

def update_dataset(lb_dataset, lb_client, s3_client, config):

    # Get list of experiments on Labelbox from existing dataset
    lb_exp_ids = set()
    lb_rows = lb_dataset.data_rows()
    for row in lb_rows:
        lb_exp_ids.add(row.external_id)

    # Get corresponding object keys in S3
    s3_object_keys = get_s3_keys(s3_client, config['s3_bucket'], config['s3_prefix'])
    s3_exp_obj_store = {}
    for k in s3_object_keys:
        curr_exp_id = re.match(r'(.*)_', Path(k).stem)[1]

        # Only add if experiment for this obj is in Labelbox dataset
        if curr_exp_id not in lb_exp_ids:
            continue

        if curr_exp_id not in s3_exp_obj_store:
            s3_exp_obj_store[curr_exp_id] = {}

        if 'movie' in k:
            s3_exp_obj_store[curr_exp_id]['movie'] = k
        elif 'mhi' in k:
            s3_exp_obj_store[curr_exp_id]['mhi'] = k
        else:
            logging.error(f"Unexpected key {k}")
            sys.exit(1)

    # Access each Labelbox data row explicitly and update URLs
    for exp_id in tqdm(s3_exp_obj_store, desc="updating rows"):
        data_row = lb_dataset.data_row_for_external_id(exp_id)

        new_movie_url = get_s3_url(s3_client, config['s3_bucket'], 
                                    s3_exp_obj_store[exp_id]['movie'],
                                    expires_in=604800)
        new_mhi_url = get_s3_url(s3_client, config['s3_bucket'],
                                    s3_exp_obj_store[exp_id]['mhi'],
                                    expires_in=604800)
        
        data_row.update(row_data=new_movie_url)
        update_lb_attachment_url(lb_client, data_row, new_mhi_url)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('config',           help="Configuration for S3 and Labelbox")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # Establish SDK clients
    
    # Labelbox
    lb_client = Client(api_key=config['LB_API_KEY'], endpoint=config['LB_ENDPOINT'])

    # S3
    aws_session = boto3.Session(profile_name=config['s3_profile'])
    s3_client = aws_session.client("s3", config=Config(s3={"use_accelerate_endpoint":True}))


    # Check if updating or uploading
    try:
        lb_ds_response = lb_client.get_datasets(where=Dataset.name == config['lb_dataset'])
        lb_existing = next(lb_ds_response)

        logging.info(f"Labelbox dataset {config['lb_dataset']} exists.")
        logging.info(f"Please confirm Dataset UID {lb_existing.uid}")
        cont = input(f"Update signed URLS for {config['lb_dataset']}? (y/n): ")

        if cont != "y":
            logging.info("Aborting.")
            sys.exit(1)
        
        update_dataset(lb_existing, lb_client, s3_client, config)

    except StopIteration:
        logging.info(f"Labelbox dataset {config['lb_dataset']} does not exist.")
        cont = input("Create dataset and continue? (y/n): ")

        if cont != "y":
            logging.info("Aborting.")
            sys.exit(1)

        upload_dataset(lb_client, s3_client, config)
