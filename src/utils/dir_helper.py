import os.path as op
import os
from pathlib import Path
import shutil
from glob import glob

from string import Template

from collections.abc import Iterable


def _get_dir(dir_key, prefix, config, rm_existing):
    '''Returns the folder associated with nested key dir_key in config. If relative, prepends prefix'''
    if not isinstance(dir_key, Iterable):
        dir_key = [dir_key]
    c = config
    for k in dir_key:
        subconfig = c
        c = c[k]
    dirpath = Template(c).substitute(subconfig)
    if not dirpath[0] == '/':
        dirpath = op.join(prefix, dirpath)

    # Delete existing tracks and plots if required
    if rm_existing:
        if op.exists(dirpath):
            shutil.rmtree(dirpath)

    if not op.exists(dirpath):
        Path(dirpath).mkdir(parents=True)
    return dirpath


def get_exp_subdir(dir_key, exp_dir, config, rm_existing=False):
    '''Returns the experiment folder associated with dir_key'''
    prefix = exp_dir

    # If config contains an experiment dir override, parse here
    # This is to allow simulatenous runs on the same experiment
    #     (e.g. multiple TOGA workers)
    # Passing different prefixes prevents clobbering output dirs,
    # but allows sharing of hologram/label dirs
    eop = 'experiment_outputs_prefix'
    if eop in config and config[eop]:
        # Don't override original data dirs
        if dir_key != "hologram_dir" and dir_key != "label_dir" and dir_key != "preproc_dir":
            prefix = op.join(config[eop], op.basename(exp_dir))
    retval = _get_dir(['experiment_dirs', dir_key], prefix, config, rm_existing)
    return retval


def get_batch_subdir(dir_key, batch_dir, config):
    '''Returns the batch folder associated with dir_key'''
    return _get_dir(['batch_dirs', dir_key], batch_dir, config, False)


def get_unique_file_by_suffix(root, suffix, logger=None):
    """
    Returns the unique file within the given root directory that has the
    specified suffix. None is returned if no such unique file exists.
    """
    candidates = glob(op.join(root, '*%s' % suffix))

    if len(candidates) == 0:
        if logger is not None:
            logger.warning(
                f'No files with suffix "*{suffix}" under "{root}"'
            )
        return None

    if len(candidates) > 1:
        if logger is not None:
            logger.warning(
                f'Multiple files with suffix "*{suffix}" under "{root}"'
            )
        return None

    return candidates[0]


def get_dir_size(folder_path):
    """
    Returns the total size of all items contained in a folder on disk.

    Parameters
    ----------
    folder_path : str
        The path to the folder to analyze.

    Returns
    -------
    total_size: int
        The total size of all items in the folder, in bytes.
    """

    total_size = 0

    for item in os.listdir(folder_path):
        item_path = op.join(folder_path, item)

        # Check if the item is a file and add it to the total size
        if op.isfile(item_path):
            total_size += op.getsize(item_path)

        # If the item is a folder, recursively call this function to get its size
        elif op.isdir(item_path):
            total_size += get_dir_size(item_path)

    return total_size