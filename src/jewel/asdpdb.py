"""
Code to manage the ASDP Database, used by JEWEL
"""
import os
import csv
from collections import defaultdict

import numpy as np
from utils.logger import get_logger
from utils.manifest import AsdpManifest

# Global variable for logging
logger = get_logger()

class DownlinkStatus:
    TRANSMITTED = 'transmitted'
    UNTRANSMITTED = 'untransmitted'


class ASDPDB:
    """
    Class providing the interface to the ASDP Database
    """

    COLUMNS = [
        'asdp_id',
        'experiment_dir',
        'manifest_file',
        'sue_file',
        'dd_file',
        'asdp_type',
        'asdp_size_bytes',
        'timestamp',
        'priority_bin',
        'downlink_status',
    ]
    ID = COLUMNS[0]

    def __init__(self, dbfile):
        self.dbfile = dbfile
        self._entries = None
        self._index = None


    def _invalidate_index(self):
        self._index = None


    def _load(self):
        if not os.path.exists(self.dbfile):
            self._entries = []
        else:
            with open(self.dbfile, 'r') as f:
                reader = csv.DictReader(f)
                self._entries = list(reader)
        self._invalidate_index()


    def _save(self):
        with open(self.dbfile, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=ASDPDB.COLUMNS)
            writer.writeheader()
            for e in self.get_entries():
                writer.writerow(e)


    def next_asdp_id(self):
        """
        Determine the next valid ASDP ID given the existing entries
        """
        ids = [int(e[ASDPDB.ID]) for e in self.get_entries()]
        return (max(ids) + 1) if len(ids) > 0 else 0


    def entry_exists(self, experiment_dir):
        """
        Check whether an entry already exists in the DB for the given
        experiment_dir

        Parameters
        ----------
        experiment_dir: str
            absolute path to the experiment directory to check for within the
            ASDPDB

        Returns
        -------
        exists: bool
            Boolean indicating whether an entry exists for this experiment
            directory
        """
        return any([
            experiment_dir == e['experiment_dir']
            for e in self.get_entries()
        ])


    def get_entries(self):
        """
        Returns a list of entries within the DB
        """
        if self._entries is None:
            self._load()
        return self._entries


    def get_entry_by_id(self, asdp_id):
        if self._index is None:
            # Rebuild Index
            entries = self.get_entries()
            self._index = {
                int(entry[ASDPDB.ID]):entry
                for entry in entries
            }
        return self._index[int(asdp_id)]


    def add_entries(self, new_entries):
        """
        Adds a list of entries to the DB

        Parameters
        ----------
        new_entries: list
            a list of dicts containing the fields for the ASDPDB entries; each
            entry should contain the columns in `ASDPDB.COLUMNS` except for the
            `ASDPDB.ID` column, which is assigned when entries are inserted into
            the database.

        Returns
        -------
        new_good_entries: list
            a list of dicts containing the entries inserted into the database
            (i.e., excluding any bad entries that could not be inserted due to
            missing columns), with populated `ASDPDB.ID` columns.
        """
        entries = self.get_entries()
        next_id = self.next_asdp_id()

        new_good_entries = []
        for e in new_entries:
            new_e = {}
            new_e[ASDPDB.ID] = next_id

            for col in (set(ASDPDB.COLUMNS) - set([ASDPDB.ID])):
                if col not in e:
                    logger.warning(
                        f'Entry {str(e)} missing column "{col}"'
                    )
                    new_e = None
                    break

                # Copy column to new entry
                new_e[col] = e[col]

            if new_e is not None:
                new_good_entries.append(new_e)
                next_id += 1

        entries.extend(new_good_entries)
        self._save()
        self._invalidate_index()

        return new_good_entries


    def set_downlink_status(self, asdp_id, downlink_status):
        """
        Sets the downlink status for an ASDP, given its ID

        Parameters
        ----------
        asdp_id: int
            the ID of an ASDP
        downlink_status: str
            the string `DownlinkStatus` representing the new status to be
            assigned to the ASDP
        """
        entry = self.get_entry_by_id(asdp_id)
        entry['downlink_status'] = downlink_status
        self._save()


    def set_priority_bin(self, asdp_id, priority_bin):
        """
        Sets the priority bin for an ASDP, given its ID

        Parameters
        ----------
        asdp_id: int
            the ID of an ASDP
        priority_bin: int
            the new downlink priority bin to which the ASDP will be assigned
        """
        entry = self.get_entry_by_id(asdp_id)
        entry['priority_bin'] = priority_bin
        self._save()


def get_timestamp(manifestfile):
    """
    Loads the timestamp of the manifest file

    Parameters
    ----------
    manifestfile: str
        path to manifest file

    Returns
    -------
    timestamp: int
        file creation time (in number of seconds since the unix epoch)
    """
    return os.path.getctime(manifestfile)


def get_path_by_entry_name(manifest, name):
    """
    Returns the absolute path within the provided manifest associated with the
    first entry with the given name

    Parameters
    ----------
    manifest: list
        list of dicts containing the manifest entries
    name: str
        name field of desired entry

    Returns
    -------
    absolute_path: str
        absolute path assocaited with the desired entry, or `None` if no such
        entry exists
    """
    for entry in manifest.entries:
        if entry['name'] == name:
            return entry['absolute_path']

    logger.warning(f'No entry with name "{name}" in manifest "{manifest}"')

    return None


def compute_asdp_size(manifest):
    """
    Computes the total size of the ASDP from the manifest entires

    Parameters
    ----------
    manifest: list
        list of dicts containing the manifest entries

    Returns
    -------
    asdp_size: int
        the total size in bytes associated with "asdp," "validate," and
        "metadata" entries in the manifest
    """
    return sum(
        int(e['filesize']) for e in manifest.entries
        if e['category'] in ('asdp', 'validate', 'metadata')
    )


def compile_asdpdb_entry(expdir, manifest_file):
    """
    Populates the columns of the ASDP DB entry

    Parameters
    ----------
    expdir: str
        path to experiment directory
    manifest_file: str
        path to manifest file associated with the experiment directory

    Returns
    -------
    entry: dict
        a dictionary containing the fields in the ASDP DB entry, or `None` if
        the fields could not be populated
    """
    entry = {}
    entry['experiment_dir'] = expdir
    entry['manifest_file'] = manifest_file

    manifest = AsdpManifest.load(manifest_file)

    entry['timestamp'] = get_timestamp(manifest_file)
    entry['sue_file'] = get_path_by_entry_name(manifest, 'science_utility')
    entry['dd_file'] = get_path_by_entry_name(manifest, 'diversity_descriptor')

    entry['asdp_type'] = manifest.asdp_type
    entry['asdp_size_bytes'] = compute_asdp_size(manifest)
    entry['priority_bin'] = manifest.priority_bin
    entry['downlink_status'] = DownlinkStatus.UNTRANSMITTED

    # Check for any missing contents
    to_check = (entry['sue_file'], entry['dd_file'], entry['asdp_type'])
    if any(e is None for e in to_check):
        logger.warning(
            f'Could not completely populate entry for {expdir}'
        )
        return None

    return entry


def load_sue(sue_file):
    """
    Loads individual SUE from CSV file

    Parameters
    ----------
    sue_file: str
        path to CSV file containing SUE

    Returns
    -------
    sue: float
        real-valued SUE entry from file, or None if an error occurs
    """

    with open(sue_file, 'r') as f:
        reader = csv.DictReader(f)
        sues = list(reader)

    if len(sues) != 1:
        logger.warning(f'Unexpected {len(sues)} != 1 entries in {sue_file}')
        return

    sue_dict = sues.pop()

    if len(sue_dict) != 1:
        logger.warning(f'Unexpected {len(sue_dict)} != 1 entries in {sue_file}')
        return

    sue = list(sue_dict.values())[0]
    return float(sue)


def load_diversity_descriptors(dd_file):
    """
    Load diversity descriptors from CSV file

    Parameters
    ----------
    dd_file: str
        path to diversity descriptors CSV file

    Returns
    -------
    (descriptors, values): (list, numpy.array)
        descriptors is a list of string descriptor names, and values is a NumPy
        array containing the floating point values associated with each; None
        is returned if an error occurs
    """

    with open(dd_file, 'r') as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    if len(entries) != 1:
        logger.warning(f'Unexpected {len(entries)} != 1 entries in {dd_file}')
        return

    dd_dict = entries.pop()

    keys = sorted(dd_dict.keys())
    values = np.array(
        [dd_dict[k] for k in keys],
        dtype=float
    )
    return keys, values


def load_entry_list_data(entry_list):
    """
    Loads the data/metadata associated with a list of ASDP DB entries

    Parameters
    ----------
    entry_list: list
        a list of ASDP DB entry dicts

    Returns
    -------
    data_dict: dict
        a dict containing the collated metadata associated with the entries in
        the list, including:
            - asdp_id: numpy.array(dtype=int)
                data product IDs
            - asdp_size_bytes: numpy.array(dtype=int)
                sizes of the data product in bytes
            - timestamp: numpy.array(float)
                timestamp of the ASDP
            - sue: numpy.array(float)
                science utility estimates for each data product
            - downlink_status: np.array(boolean)
                whether each data product has been downlinked
            - dd: np.array(float)
                n-by-k array of k diversity descriptors for each of the n
                entries
            - dd_features: list(str)
                the str names of the k diversity descriptors
    """

    entry_data = defaultdict(list)
    last_keys = None

    for e in entry_list:
        eid = int(e[ASDPDB.ID])
        size = int(e['asdp_size_bytes'])
        ts = float(e['timestamp'])
        is_not_downlinked = (
            e['downlink_status'] == DownlinkStatus.UNTRANSMITTED
        )

        sue = load_sue(e['sue_file'])
        if sue is None:
            logger.warning(f'Error loading SUE; skipping entry {eid}')
            continue

        dd_items = load_diversity_descriptors(e['dd_file'])
        if dd_items is None:
            logger.warning(f'Error loading DD; skipping entry {eid}')
            continue
        dd_keys, dd_values = dd_items

        if not ((last_keys is None) or (dd_keys == last_keys)):
            logger.warning(f'DD key mismatch; skipping entry {eid}')
            continue

        # Add entries
        entry_data[ASDPDB.ID].append(eid)
        entry_data['asdp_size_bytes'].append(size)
        entry_data['timestamp'].append(ts)
        entry_data['sue'].append(sue)
        entry_data['dd'].append(dd_values)
        entry_data['downlink_status'].append(is_not_downlinked)

        last_keys = dd_keys

    data_dict = dict([(k, np.array(v)) for k, v in entry_data.items()])
    if last_keys is not None:
        data_dict['dd_features'] = last_keys

    return data_dict


def load_asdp_metadata_by_bin_and_type(asdpdb):
    """
    Loads entry list metadata for each ASDP bin and type

    Parameters
    ----------
    asdpdb: ASDPDB
        an ASDP DB instance

    Returns
    -------
    metadata_dict: dict
        each nested dict entry is keyed by ASDP bin and then type, and each
        value is a dict containing the metadata associated with the type entry
        list (see `load_entry_list_data`)
    """
    entries_by_type_and_bin = defaultdict(lambda: defaultdict(list))
    for entry in asdpdb.get_entries():
        pbin = int(entry['priority_bin'])
        atype = entry['asdp_type']
        entries_by_type_and_bin[pbin][atype].append(entry)

    return {
        b:{
            t:load_entry_list_data(l)
            for t, l in tdict.items()
        }
        for b, tdict in entries_by_type_and_bin.items()
    }


def save_asdp_ordering(outputfile, ordering):
    """
    Saves an ASDP ordering produced by JEWEL to a CSV file

    Parameters
    ----------
    outputfile: str
        path to output CSV file
    ordering: list
        list of dicts containing entries for the following fields:
            - asdp_id
            - initial_sue
            - final_sue
            - initial_sue_per_byte
            - final_sue_per_byte
            - size_bytes
            - timestamp
    """
    with open(outputfile, 'w') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                'asdp_id', 'initial_sue', 'final_sue',
                'initial_sue_per_byte', 'final_sue_per_byte',
                'size_bytes', 'timestamp',
            ),
        )
        writer.writeheader()
        for entry in ordering:
            writer.writerow(entry)


def load_asdp_ordering(orderfile):
    """
    Loads an ASDP ordering produced by JEWEL from a CSV file

    Parameters
    ----------
    orderfile: str
        path to order CSV file

    Returns
    -------
    ordering: list
        list of dicts containing entries for the following fields:
            - asdp_id
            - initial_sue
            - final_sue
            - initial_sue_per_byte
            - final_sue_per_byte
            - size_bytes
            - timestamp
    """
    with open(orderfile, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)
