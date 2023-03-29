'''
Utility functions for generating asdp manifests
'''
import sys
import os
import os.path as op
import json
import yaml
import logging
from collections import OrderedDict

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
    """ Returns the cumulative filesize of files in a directory (does not
    include any files nested in subdirectories)

    Parameters
    ----------
    path: string
        Path to a directory
    
    Returns
    -------
    cum_filesize: int
        Cumulative filesize in bytes. Returns 0 if no files exist, or the
        directory does not exist.
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


def load_manifest_metadata(metadata_file, metadata_str):
    """
    Loads the manifest metadata from a file and raw string; if multiple value
    are provied for the same key, the raw string takes precedence

    Parameters
    ----------
    metadata_file: str
        path to a YAML metadata file containing dict, or None
    metadata_str: str
        raw YAML metadata dict string, or None

    Returns
    -------
    manifest_metadata: dict
        manifest metadata dictionary
    """
    manifest_metadata = {}

    if metadata_file is not None:
        with open(metadata_file, 'r') as f:
            metadata = yaml.safe_load(f)
        manifest_metadata.update(metadata)

    if metadata_str is not None:
        metadata = yaml.safe_load(metadata_str)

        # Add warning for any keys that will be overridden
        replaced_keys = [
            k for k in metadata.keys()
            if k in manifest_metadata
        ]
        if len(replaced_keys):
            logging.warning(f'The following manifest metadata keys will be overridden: {replaced_keys}')

        manifest_metadata.update(metadata)

    return manifest_metadata


class AsdpManifest:


    def __init__(self, asdp_type, priority_bin,
            root_dir=None, total_size=0, entries=None, metadata=None):
        """
        Initialize ASDP Manifest

        Parameters
        ----------
        asdp_type: str
            ASDP type (e.g., HELM, ACME)
        priority_bin: int
            ASDP Priority Bin
        root_dir: str
            ASDP root directory, or None
        total_size: int
            Total size in bytes (optional)
        entries: list
            List of tuples containing (name, category, absolute_path, size),
            if loading from a saved manifest (optional)
        metadata: dict
            Optional metadata dict (if loading from saved manifest)
        """

        self.asdp_type = asdp_type
        self.priority_bin = priority_bin
        self._root_dir = root_dir
        self.total_size = total_size
        self._entries = [] if entries is None else entries
        self.metadata = {} if metadata is None else metadata


    @property
    def root_dir(self):
        """
        Calculates or returns the ASDP root directory
        """

        # If a path has been provided, return it
        if self._root_dir is not None:
            return self._root_dir

        # Otherwise, collect all absolute paths
        absolute_paths = [
            e['absolute_path'] for e in self._entries
            if 'absolute_path' in e
        ]
        n_paths = len(absolute_paths)
        if n_paths == 0:
            # If no absolute paths, we do not have a root directory
            return None
        elif n_paths == 1:
            # If only one entry, its directory is the root
            return op.dirname(absolute_paths[0])
        else:
            # Otherwise, the longest common prefix is the root
            return op.commonpath(absolute_paths)


    @property
    def entries(self):
        """
        Returns a set of entries with default attributes
        """
        return self.get_entries()


    def get_entries(self, include_absolute=True, include_relative=True):
        """
        The the list of ASDP entries, potentially excluding certain path
        entries

        Parameters
        ----------
        include_absolute: bool (default: True)
            Include the absolute path to the ASDP entry
        include_relative: bool (default: True)
            Include the relative path to the ASDP entry

        Returns
        -------
        list of entries, with absolute and relative paths included as specified
        """
        root = self.root_dir
        full_entries = [dict(e) for e in self._entries]
        for e in full_entries:
            if 'absolute_path' in e:
                e['relative_path'] = op.relpath(e['absolute_path'], root)
            elif 'relative_path' in e:
                rel = e['relative_path']
                if root is None:
                    e['absolute_path'] = rel
                else:
                    e['absolute_path'] = op.join(root, rel)

        if not include_absolute:
            for e in full_entries:
                del e['absolute_path']

        if not include_relative:
            for e in full_entries:
                del e['relative_path']

        return full_entries

    def add_entry(self, name, category, absolute_path):
        """
        Add ASDP Manifest File/Directory Entry

        Parameters
        ----------
        name: str
            ASDP Entry Name
        category: str
            ASDP Entry Category (e.g., validate, predict, asdp, metadata)
        absolute_path: str
            Absolute path to the ASDP entry (file or directory)
        """

        if op.isdir(absolute_path):
            size = get_dirsize(absolute_path)
        elif op.isfile(absolute_path):
            size = get_filesize(absolute_path)
        else:
            size = 0

        # Update total size
        self.total_size += size

        self._entries.append({
            'name': name,
            'category': category,
            'absolute_path': absolute_path,
            'filesize': size,
        })


    def add_metadata(self, **metadata):
        """
        Add metadata entries to manifest

        Parameters
        ----------
        metadata: dict (**kwargs)
            kwargs or dict containing metadata entries
        """
        self.metadata.update(metadata)


    def write(self, out_path):
        """
        Writes ASDP product manifest to the specified out_path.

        Parameters
        ----------
        out_path: string
            Output path for the manifest. Should be a .json
        """

        # Ensure output path ends in ".json"
        outname, outext = op.splitext(out_path)
        if outext != '.json':
            logging.warning("File extension for manifest output is not json, replacing.")
            out_path = outname + '.json'

        # Construct output dict
        out_data = OrderedDict([
            ('type', self.asdp_type),
            ('priority_bin', self.priority_bin),
            ('root_directory', self.root_dir),
            ('total_size', self.total_size),
            ('entries', self.get_entries(include_absolute=False)),
            ('metadata', self.metadata),
        ])

        with open(out_path, 'w') as f:
            json.dump(out_data, f, indent=2)


    def relocated_path(self, experiment_dir, entry):
        """
        Updates relative manifest file paths for relocated experiment
        directories. This function hard-codes the relationship between the
        experiment directory and the manifest file location for each type.

        Parameters
        ----------
        experiment_dir: str
            new experiment directory path
        entry: dict
            manifest file entry

        Returns
        -------
        absolute path of the relocated manifest entry
        """

        if self.asdp_type in ('helm', 'fame',):
            return op.normpath(
                op.join(experiment_dir, '..', entry['relative_path'])
            )

        elif self.asdp_type in ('hirails',):
            return op.normpath(
                op.join(experiment_dir, '..', '..', entry['relative_path'])
            )

        elif self.asdp_type in ('acme',):
            return op.join(experiment_dir, entry['relative_path'])

        else:
            logging.warning(f'Unsupported type {self.asdp_type}')
            return op.join(experiment_dir, entry['relative_path'])


    @staticmethod
    def load(manifestfile):
        """
        Loads a manifest file

        Parameters
        ----------
        manifestfile: str
            path to manifest file
        """
        with open(manifestfile, 'r') as f:
            data = json.load(f)
            return AsdpManifest(
                asdp_type=data['type'],
                priority_bin=data['priority_bin'],
                root_dir=data.get('root_directory', None),
                total_size=data['total_size'],
                entries=data['entries'],
                metadata=data['metadata'],
            )
