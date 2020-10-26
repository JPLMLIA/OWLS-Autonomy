import os.path as op
import os
from pathlib import Path
import shutil

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
        if dir_key != "hologram_dir" and dir_key != "label_dir":
            prefix = op.join(config[eop], op.basename(exp_dir))
    retval = _get_dir(['experiment_dirs', dir_key], prefix, config, rm_existing)
    return retval
    
def get_batch_subdir(dir_key, batch_dir, config):
    '''Returns the batch folder associated with dir_key'''
    return _get_dir(['batch_dirs', dir_key], batch_dir, config, False)