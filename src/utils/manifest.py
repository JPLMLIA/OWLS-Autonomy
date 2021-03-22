'''
Utility functions for generating asdp manifests
'''
import sys
import os
import os.path as op
import csv
import logging

def get_filesize(path):
    """ Returns the filesize of a file 
    
    Parameters
    ----------
    path: string
        Absolute or relative path to a file
    
    Returns
    -------
    Filesize in bytes, integer. 0 if file does not exist.
    
    """

    if op.isfile(path):
        return op.getsize(path)
    else:
        return 0

def get_dirsize(path):
    """ Returns the cumulative filesize of files in a directory

    Parameters
    ----------
    path: string
        Path to a directory
    
    Returns
    -------
    cum_filesize: int
        Cumulative filesize in bytes. Returns 0 if no files exist, or the directory does not exist.
    """

    if not op.isdir(path):
        return 0

    files = os.listdir(path)
    cum_filesize = 0
    for f in files:
        fp = op.join(path, f)
        if op.isfile(fp):
            cum_filesize += op.getsize(fp)
    
    return cum_filesize


def write_manifest(asdp_list, out_path):
    """ Writes a ASDP product manifest to the specified out_path.

    The written CSV has the following columns:
    absolute_path:  absolute path to the product
    name:           name of the product
    type:           type of the product (e.g., ACME or HELM)
    category:       category of product, if it exists
    filesize:       filesize of specified product

    Parameters
    ----------
    asdp_list: list of tuples
        List of tuples (abs_path, name, type, category), (string, string, string, string)
        abs_path can be a path to a file or directory. Only files in the first level of the directory will be included in the manifest.
    out_path: string
        Output path for the manifest. Should be a .csv
    """

    # Force manifest output to be a CSV
    outname, outext = op.splitext(out_path)
    if outext != '.csv':
        logging.warning("File extension for manifest output is not csv, replacing.")
        out_path = outname + '.csv'
    
    # Get sizes and write to CSV
    with open(out_path, 'w') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['absolute_path', 'name', 'type', 'category', 'filesize'])

        # Iterate through asdp list
        for abs_path, name, ptype, category in asdp_list:
            if op.isdir(abs_path):
                size = get_dirsize(abs_path)
            elif op.isfile(abs_path):
                size = get_filesize(abs_path)
            else:
                size = 0
            
            writer.writerow([abs_path, name, ptype, category, size])
    
