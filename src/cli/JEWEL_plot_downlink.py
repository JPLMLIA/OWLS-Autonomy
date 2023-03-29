#!/usr/bin/env python
import os
import re
import csv
import json
import base64
import argparse
import os.path as op
from glob import glob
from io import BytesIO
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
from jinja2 import FileSystemLoader, Environment
from sklearn.manifold import MDS
from collections import defaultdict
import matplotlib.pyplot as plt

from bokeh.models import (
    HoverTool, ColumnDataSource, TapTool, OpenURL,
    CategoricalColorMapper, Row, DataTable, TableColumn
)
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import Colorblind

from fsw.JEWEL.asdpdb import (
    load_asdp_ordering, ASDPDB,
    load_diversity_descriptors,
    load_dqe as load_data_quality_estimate
)
from utils.manifest import AsdpManifest

TEMPLATE_FILE = 'downlink_template.html.j2'
ACME_TEMPLATE_FILE = 'acme_details_template.html.j2'
TEMPLATE_DIR = 'templates'
ASDPDB_FILE = 'asdpdb.csv'
ORDER_FILE = 'ordering.csv'
HTML_FILE = 'index.html'
DETAILS_DIR = 'details'

SCRIPT_DIR = op.dirname(op.realpath(__file__))


def get_asdp_path(manifest, entry_name, force_unique=True):
    """
    Extracts the absolute path to the manifest entry

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP
    entry_name: str
        load path associated with this ASDP entry name
    force_unique: bool (default: True)
        ensure the entry specified by name is unique

    Returns
    -------
    absolute path to the first manifest entry matching the given name
    """
    paths = [
        e['relative_path'] for e in manifest.entries
        if e['name'] == entry_name
    ]
    if force_unique:
        assert len(paths) == 1
    path = paths[0]
    return op.join(manifest.asdpdir, path)


def load_manifest(asdpdir):
    """
    Loads the manifest file (unique file ending with "manifest.json") within
    an ASDP directory

    Parameters
    ----------
    asdpdir: str
        path to ASDP directory

    Returns
    -------
    unique AsdpManifest object for manifest within the directory
    """
    # Get manifest file
    manifest_candidates = glob(op.join(asdpdir, '*manifest.json'))
    assert len(manifest_candidates) == 1
    manifest_file = manifest_candidates.pop()
    manifest = AsdpManifest.load(manifest_file)

    # Assign asdpdir to manifest
    manifest.asdpdir = asdpdir

    return manifest


def get_asdp_dir(sessiondir, asdp_id):
    """
    Get the ASDP directory within a downlink session directory

    Parameters
    ----------
    sessiondir: str
        downlink session root directory
    asdp_id: str
        unique ASDP identifier

    Returns
    -------
    absolute path to ASDP directory with associated ID
    """
    return op.join(sessiondir, 'asdp%09d' % int(asdp_id))


def load_dd(manifest):
    """
    Load diversity descriptor associated with an AsdpManifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    dict mapping diversity descriptor names to numerical values
    """
    dd_path = get_asdp_path(manifest, 'diversity_descriptor')
    dds = load_diversity_descriptors(dd_path)
    return dict(zip(*dds))


def load_dqe(manifest):
    """
    Loads the data quality estimate (DQE) associated with an AsdpManifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    data quality estimate, or 1.0 if no data quality estimate was found
    """
    try:
        dqe_path = get_asdp_path(manifest, 'data_quality')
    except AssertionError:
        # If DQE not present, just return 1.0
        return 1.0
    dqe = load_data_quality_estimate(dqe_path)
    return dqe


def load_tracks(manifest):
    """
    Loads the track data associated with an AsdpManifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    list of parsed JSON track files
    """
    trackdir = get_asdp_path(manifest, 'predicted_tracks')
    jsonfiles = glob(op.join(trackdir, '*.json'))
    tracks = []
    for f in jsonfiles:
        with open(f, 'r') as f:
            tracks.append(json.load(f))
    return tracks


def compute_track_stats(tracks):
    """
    Computes class statistics for a list of tracks

    Parameters
    ----------
    tracks: list
        list of parsed JSON tracks

    Returns
    -------
    dict mapping class names to counts across all tracks
    """
    stats = defaultdict(int)
    for track in tracks:
        cls = track['classification']
        stats[cls] += 1
    return stats


def make_track_visualization(tracks, size=(1024, 1024)):
    """
    Generates base64 encoded visualization of all tracks, with motile tracks
    shown in magenta and other tracks shown in cyan

    Parameters
    ----------
    tracks: list
        list of parsed JSON tracks
    size: tuple
        tuple of int values (rows, columns) indicating the size of the
        observation on which to draw tracks

    Returns
    -------
    dict with a base64 color PNG image of the specified size with a black
    background and traces color coded according to class label under the key
    `track_visualization`
    """
    im = Image.new('RGB', size)
    draw = ImageDraw.Draw(im)

    paint_order = ['other', 'non-motile', 'motile']
    stracks = sorted(
        tracks,
        key=lambda t: paint_order.index(t['classification'])
    )

    for track in stracks:
        positions = [
            tuple(xy[::-1]) for xy in track['Particles_Position']
            if xy is not None
        ]
        motile = (track['classification'] == 'motile')
        if motile:
            width = 5
            fill = 'magenta'
        else:
            width = 1
            fill = 'cyan'
        draw.line(positions, fill=fill, width=width, joint='curve')

    io = BytesIO()
    im.save(io, "PNG")
    io.seek(0)
    encoded = base64.b64encode(io.getvalue()).decode('ascii')
    return {'track_visualization': encoded}


def load_mhi(manifest):
    """
    Loads the motion history image (MHI) as a base64 string

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    dict with a base64 encoded MHI image associated with key `mhi_b64`
    """
    mhifile = get_asdp_path(manifest, 'mhi_image_info')
    io = BytesIO()
    with Image.open(mhifile) as im:
        im.convert("RGB").save(io, "JPEG")
    io.seek(0)
    encoded = base64.b64encode(io.getvalue()).decode('ascii')
    return {'mhi_b64': encoded}


def get_track_data(tracks):
    """Loads track data into a dict of lists

    Parameters
    ----------
    tracks: list
        list of parsed tracks from JSON files

    Returns
    -------
    dict containing a `track_data` entry, which maps to a dict containing:
        `ids`: track ids
        `probabilities`: motility probabilities
        `start_times`: track start times
        `end_times`: track end times
        `xs`, `ys`: particle positions (lists)
    """
    data = defaultdict(list)

    for t in tracks:
        data['ids'].append(t['Track_ID'])
        data['probabilities'].append(t['probability_motility'])

        times = t['Times']
        data['start_times'].append(min(times))
        data['end_times'].append(max(times))

        positions = t['Particles_Position']
        xs = [p[1] for p in positions if p is not None]
        ys = [p[0] for p in positions if p is not None]
        data['xs'].append(xs)
        data['ys'].append(ys)

    return {'track_data': data}


def load_helm_stats(manifest):
    """
    Loads statistics and auxiliary information associated with a HELM manifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    dict containing track class statistics, a track visualization, and motion
    history image
    """
    stats = {}

    tracks = load_tracks(manifest)
    track_stats = compute_track_stats(tracks)
    stats.update(track_stats)

    track_data = get_track_data(tracks)
    stats.update(track_data)

    track_viz = make_track_visualization(tracks)
    stats.update(track_viz)

    mhi = load_mhi(manifest)
    stats.update(mhi)

    return stats


def load_tic(manifest):
    """
    Load total ion count (TIC) base64-encoded visualization associated with an
    ACME manifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    dict containing a base64 encoded total ion count visualization under the
    key `tic_b64`
    """
    ticfile = get_asdp_path(manifest, 'total_ion_count')
    with open(ticfile, 'r') as f:
        data = np.array(f.read().strip().split(','), dtype=int)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.plot(data, 'w-')
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)

    ax.tick_params(axis='both', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    io = BytesIO()
    plt.savefig(io,
        format='jpg', facecolor='black',
        dpi=100, bbox_inches='tight'
    )
    io.seek(0)
    encoded = base64.b64encode(io.getvalue()).decode('ascii')

    plt.close()

    return {'tic_b64': encoded}


def load_peak_properties(manifest):
    """
    Load a base64-encoded visualization of peak properties (mass and charge)
    associated with an ACME manifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    dict containing a base64 encoded peak property visualization under the key
    `peaks`
    """

    peakfile = get_asdp_path(manifest, 'peak_properties')
    with open(peakfile, 'r') as f:
        reader = csv.DictReader(f)
        properties = list(reader)

    xs = np.array([
        p['Peak Central Time (Min)']
        for p in properties
    ], dtype=float)
    ys = np.array([
        p['Mass (amu)']
        for p in properties
    ], dtype=float)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.plot(xs, ys, 'w.')
    ax.set_xlabel('Time (min)', fontsize=14)
    ax.set_ylabel('Mass (amu)', fontsize=14)
    _, ymax = ax.get_ylim()
    ax.set_ylim(ymax, 0)

    ax.tick_params(axis='both', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    io = BytesIO()
    plt.savefig(io,
        format='jpg', facecolor='black',
        dpi=100, bbox_inches='tight'
    )
    io.seek(0)
    encoded = base64.b64encode(io.getvalue()).decode('ascii')

    plt.close()

    return {
        'peaks': encoded,
        'peak_properties': properties,
    }


def load_peak_mugshot_data(mugshot_dir, peak):
    """
    Load the total counts curve from a peak mugshot

    Parameters
    ----------
    mugshot_dir: str
        ASDP mugshot directory
    peak: dict
        dictionary of peak properties

    Returns
    -------
    array of total counts (sum across mass) over time for the specified peak
    """

    peak_mass = round(float(peak['Mass (amu)']), 2)
    peak_time = round(float(peak['Peak Central Time (Min)']), 2)
    mugshot_paths = glob(op.join(mugshot_dir,
        f'Time_Mass_Max_{peak_time:.2f}_{peak_mass:.2f}_*.tif'
    ))
    if len(mugshot_paths) == 0:
        raise ValueError(
            f'No mugshot found for peak with mass = {peak_mass}, time = {peak_time}'
        )
    if len(mugshot_paths) > 1:
        print(f'Warning: multiple mugshots for peak with mass = {peak_mass}, time = {peak_time}')

    mugshot_path = mugshot_paths[0]

    mugshot_file_re = 'Time_Mass_Max_([^_]*)_([^_]*)_([^_]*).tif'
    match = re.match(mugshot_file_re, op.basename(mugshot_path))
    if match is None:
        raise ValueError(f'Unexpected ACME mugshot file name: {b}')
    max_count = float(match.group(3))

    with Image.open(mugshot_path) as im:
        I = np.array(im)

    I_scaled = np.round((I * max_count) / 255.)
    total_count = np.sum(I_scaled, axis=0)

    return total_count


def generate_peak_mugshot_plot(peak, total_count):
    """
    Generate base64-encoded peak total count plot

    Parameters
    ----------
    peak: dict
        peak properties
    total_count: numpy.array
        array of total counts at each time from peak mugshot

    Returns
    -------
    base64 JPG image of peak plot
    """

    peak_time = round(float(peak['Peak Central Time (Min)']), 2)
    left = float(peak['Peak Left Time (Min)'])
    right = float(peak['Peak Right Time (Min)'])
    times = np.linspace(left, right, len(total_count))

    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.plot(times, total_count, 'w-')
    ax.set_xlabel('Time (Min)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_xticks([left, peak_time, right])
    ax.tick_params(axis='both', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    io = BytesIO()
    plt.savefig(io,
        format='jpg', facecolor='black',
        dpi=100, bbox_inches='tight'
    )
    io.seek(0)
    encoded = base64.b64encode(io.getvalue()).decode('ascii')

    plt.close()

    return encoded


def load_peak_mugshots(manifest, peak_properties):
    """
    Load peak mugshots total counts for each peak

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP
    peak_properties: list
        list of dictionaries containing peak properties

    Returns
    -------
    dict containing list of mugshots, one for each peak
    """

    mugshot_dir = get_asdp_path(manifest, 'peak_mugshots')

    mugshots = [
        load_peak_mugshot_data(mugshot_dir, peak)
        for peak in peak_properties
    ]

    return { 'mugshots': mugshots }


def load_acme_stats(manifest):
    """
    Loads statistics and auxiliary information associated with an ACME manifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    dict containing peak properties and total ion count visualizations
    """
    stats = {}

    properties = load_peak_properties(manifest)
    stats.update(properties)

    mugshots = load_peak_mugshots(manifest, properties['peak_properties'])
    stats.update(mugshots)

    tic = load_tic(manifest)
    stats.update(tic)

    return stats


def load_stats(manifest):
    """
    Loads statistics and auxiliary information associated with an ASDP manifest

    Parameters
    ----------
    manifest: AsdpManifest
        manifest object for an ASDP

    Returns
    -------
    dict containing statistics loaded for the ASDP
    """
    t = manifest.asdp_type
    if t == 'helm':
        load_f = load_helm_stats
    elif t == 'fame':
        # Re-use HELM loader
        load_f = load_helm_stats
    elif t == 'acme':
        load_f = load_acme_stats
    else:
        raise ValueError('Unknown ASDP type "%s"' % t)

    return load_f(manifest)


def load_downlink_info(sessiondir):
    """
    Load ASDP information from downlink session directory

    Parameters
    ----------
    sessiondir: str
        downlink session directory

    Returns
    -------
    list of dicts containing information associated with each ASDP in downlink
    prioritization order
    """
    asdpdb_file = op.join(sessiondir, ASDPDB_FILE)
    ordering_file = op.join(sessiondir, ORDER_FILE)

    db = ASDPDB(asdpdb_file)
    ordering = load_asdp_ordering(ordering_file)

    for i, o in tqdm(list(enumerate(ordering, 1)), 'Loading Downlink Info'):
        asdp_id = o['asdp_id']

        o['rank'] = i

        entry = db.get_entry_by_id(asdp_id)
        o['asdp_db_entry'] = entry

        asdpdir = get_asdp_dir(sessiondir, asdp_id)
        manifest = load_manifest(asdpdir)

        o['asdp_local_dir'] = op.abspath(asdpdir)
        o['asdp_onboard_dir'] = op.abspath(manifest.root_dir)

        dd = load_dd(manifest)
        o['dd'] = dd

        dqe = load_dqe(manifest)
        o['dqe'] = dqe

        stats = load_stats(manifest)
        o['stats'] = stats

    return ordering


def get_mds_vectors(id_dd_list):
    """
    Perform multi-dimensional scaling (MDS) dimensionality reduction on
    diversity descriptor (DD) vectors to produce a 2-dimensional representation
    for each ASDP

    Parameters
    ----------
    id_dd_list: list
        list of tuples containing integer ASDP IDs DD vectors

    Returns
    -------
    dict mapping ASDP ID to 2-dimensional DD components
    """
    ids, dd_dicts = zip(*id_dd_list)

    # Get unique DDs
    key_sets = set([
        frozenset(d.keys()) for d in dd_dicts
    ])
    assert len(key_sets) == 1
    keys = sorted(key_sets.pop())
    dds = np.array([
        [d[k] for k in keys]
        for d in dd_dicts
    ])

    mds = MDS(n_components=2, random_state=0)
    components = mds.fit_transform(dds)

    mds_dict = dict([
        (i, c) for i, c in zip(ids, components)
    ])
    return mds_dict


def get_dd_mds_vectors(downlink_info):
    """
    Get reduced dimensionality diversity descriptor (DD) vectors for each ASDP
    type

    Parameters
    ----------
    downlink_info: dict
        downlinked ASDP information as returned by `load_downlink_info`

    Returns
    -------
    tuple containing list of asdp_types, and dict containing `x` and `y`
    components for each DD representation vector
    """
    dd_lists = defaultdict(list)
    for i in downlink_info:
        asdp_type = i['asdp_db_entry']['asdp_type']
        dd_lists[asdp_type].append(
            (int(i['asdp_id']), i['dd'])
        )

    asdp_ids = [
        int(i['asdp_id']) for i in downlink_info
    ]
    asdp_types = list(dd_lists.keys())

    mds_dicts = dict([
        (t, get_mds_vectors(l))
        for t, l in dd_lists.items()
    ])

    data_dict = {}

    for t in asdp_types:
        mds = np.array([
            mds_dicts[t].get(i, (np.nan, np.nan))
            for i in asdp_ids
        ])
        data_dict['%s_x' % t] = mds[:, 0].tolist()
        data_dict['%s_y' % t] = mds[:, 1].tolist()

    return asdp_types, data_dict


def create_helm_details_page(asdp_info, details_info):
    """Creates a details page for a HELM/FAME ASDP

    Parameters
    ----------
    asdp_info: dict
        dictionary containing information for the ASDP
    details_info: dict
        dictionary containing information for the overall details pages

    Returns
    -------
    Bokeh `html` object representing details page
    """

    data = asdp_info['stats']['track_data']

    source = ColumnDataSource(data=data)

    p1 = figure(title="Tracks",
        plot_width=500, plot_height=500,
        tools="tap,pan,box_zoom,wheel_zoom,save,reset"
    )
    p1.multi_line('xs', 'ys', line_width=2, source=source)
    p1.y_range.flipped = True

    columns = [
            TableColumn(field="ids", title="Track ID"),
            TableColumn(field="start_times", title="Track Start Time"),
            TableColumn(field="end_times", title="Track End Time"),
            TableColumn(field="probabilities", title="p(motile)")
        ]
    dt1 = DataTable(source=source, columns=columns, width=500, height=500)

    plot = Row(p1, dt1)

    html = file_html(plot,
        title="JEWEL Downlink Visualization",
        resources=CDN,
    )

    return html


def create_acme_details_page(asdp_info, details_info):
    """Creates a details page for a ACME ASDP

    Parameters
    ----------
    asdp_info: dict
        dictionary containing information for the ASDP
    details_info: dict
        dictionary containing information for the overall details pages

    Returns
    -------
    Bokeh `html` object representing details page
    """

    column_names = [
        ("Peak Central Time (Min)", "Peak Central Time (Min)",),
        ("Mass (amu)",              "Mass (amu)",),
        ("Peak Volume (Counts)",    "Peak Volume (Counts)",),
        ("Peak Amplitude (Counts)", "Peak Amplitude (Counts)",),
        ("Peak Amplitude (ZScore)", "Peak Amplitude (ZScore)",),
        ("Peak Base Width (sec)",   "Peak Base Width (sec)",),
    ]

    # Get peak properties in list format
    peaks = asdp_info['stats']['peak_properties']
    peak_data = defaultdict(list)
    for i, entry in enumerate(peaks):
        peak_data['url'].append('#peak_info_%d' % i)
        for k, v in entry.items():
            peak_data[k].append(v)

    peak_data = {
        k : np.array(v, dtype=float) if k != 'url' else v
        for k, v in peak_data.items()
    }

    source = ColumnDataSource(data=peak_data)

    pp = figure(title="Peaks",
        plot_width=500, plot_height=500,
        tools="tap,pan,box_zoom,wheel_zoom,save,reset"
    )
    pp.scatter('Peak Central Time (Min)', 'Mass (amu)', source=source,
        size=7)
    pp.xaxis.axis_label = 'Peak Central Time (Min)'
    pp.yaxis.axis_label = 'Mass (amu)'

    taptool = pp.select(type=TapTool)
    taptool.callback = OpenURL(url="@url", same_tab=True)

    columns = [
        TableColumn(field=f, title=t)
        for f, t in column_names
    ]

    table = DataTable(source=source, columns=columns, width=1024, height=500)

    plot = Row(pp, table)

    mugshots = [
        generate_peak_mugshot_plot(peak, mugshot)
        for peak, mugshot in zip(peaks, asdp_info['stats']['mugshots'])
    ]

    template_variables = {
        'peaks': list(zip(
            range(len(peaks)),
            peaks,
            mugshots
        )),
    }

    html = file_html(plot,
        title="ACME Details Visualization",
        resources=CDN,
        template=details_info['acme_template'],
        template_variables=template_variables,
    )

    return html


def create_detail_page(detailsdir, asdp_info, details_info):
    """Creates a details page for an ASDP

    Parameters
    ----------
    detailsdir: str
        directory for writing details HTML pages
    asdp_info: dict
        dictionary containing information for the ASDP
    details_info: dict
        dictionary containing information for the overall details pages
    """

    asdp_type = asdp_info['asdp_db_entry']['asdp_type']

    if asdp_type in ('helm', 'fame'):
        html = create_helm_details_page(asdp_info, details_info)

    elif asdp_type in ('acme',):
        html = create_acme_details_page(asdp_info, details_info)

    else:
        # Not Supported
        return

    asdp_id = int(asdp_info['asdp_id'])
    outputfile = op.join(detailsdir, ('asdp%09d.html' % asdp_id))
    with open(outputfile, 'w') as f:
        f.write(html)


def plot_downlink(sessiondir, outputdir):
    """Main downlink plot function

    Parameters
    ----------
    sessiondir: str
        downlink session directory
    outputdir: str
        HTML output directory
    """

    # Load jinja2 template for plot
    # See https://docs.bokeh.org/en/latest/docs/user_guide/embed.html
    # for more information about extending the base bokeh template
    templateLoader = FileSystemLoader(
        searchpath=op.join(SCRIPT_DIR, TEMPLATE_DIR)
    )
    templateEnv = Environment(loader=templateLoader)
    template = templateEnv.get_template(TEMPLATE_FILE)
    acme_template = templateEnv.get_template(ACME_TEMPLATE_FILE)

    # Load ASDP downlink info
    downlink_info = load_downlink_info(sessiondir)

    # Get DD MDS vectors by type
    asdp_types, dd_mds_vectors = get_dd_mds_vectors(downlink_info)

    cf = figure(
        title='Cumulative Marginal SUE',
        plot_width=500, plot_height=500
    )

    # Setup data for Bokeh column format
    data = {
        'xs': [],
        'ys': [],
        'asdp_id': [],
        'asdp_type': [],
        'init_sue': [],
        'final_sue': [],
        'sue_per_byte': [],
        'dqe': [],
        'url': [],
        'cumulative_downlink': [],
    }
    data.update(dd_mds_vectors)

    for t in asdp_types:
        data['cumulative_sue_%s' % t] = []

    # Add ASDP information to Bokeh column data structure
    total_dv = 0
    total_sue = 0
    total_sue_by_type = defaultdict(float)
    for i in downlink_info:
        size = int(i['size_bytes'])
        sue = float(i['final_sue'])
        init_sue = float(i['initial_sue'])
        asdp_type = i['asdp_db_entry']['asdp_type']
        xstart = total_dv
        xend = total_dv + size
        ystart = total_sue
        yend = total_sue + sue

        x = [xstart, xstart, xend, xend]
        y = [ystart, 0, 0, yend]

        data['xs'].append(x)
        data['ys'].append(y)
        data['asdp_id'].append(int(i['asdp_id']))
        data['asdp_type'].append(asdp_type.upper())
        data['init_sue'].append(init_sue)
        data['final_sue'].append(sue)
        data['sue_per_byte'].append(float(i['final_sue_per_byte']))
        data['dqe'].append(i.get('dqe', 1.0))
        data['url'].append('#asdp_info_%d' % int(i['asdp_id']))

        total_sue_by_type[asdp_type] += sue
        for t in asdp_types:
            data['cumulative_sue_%s' % t].append(total_sue_by_type[t])

        total_dv += size
        total_sue += sue

        data['cumulative_downlink'].append(total_dv)

    # Normalize cumulative SUE
    for t in asdp_types:
        key = 'cumulative_sue_%s' % t
        cumulative_sue = np.array(data[key])
        data[key] = cumulative_sue / np.max(cumulative_sue)

    # Create a Bokeh data source
    source = ColumnDataSource(data=data)

    # Construct color map to distinguish each ASDP type
    all_types = sorted(set(data['asdp_type']))
    palette = Colorblind[max(3, len(all_types))]
    cmap = CategoricalColorMapper(factors=all_types, palette=palette)

    # Add cumulative SUE plot
    cf.patches(xs='xs', ys='ys',
        line_color='black',
        fill_color={'field': 'asdp_type', 'transform': cmap},
        alpha=0.5,
        hover_fill_color='red',
        hover_alpha=1.0,
        source=source)

    cf.xaxis.axis_label = 'Downlink Data Volume'
    cf.yaxis.axis_label = 'Cumulative Marginal SUE'

    # Setup hover tool to link plots
    hover = HoverTool(
        tooltips=[
            ('ASDP ID', '@asdp_id'),
            ('ASDP Type', '@asdp_type'),
            ('Initial SUE', '@init_sue{0.3f}'),
            ('Final SUE', '@final_sue{0.3f}'),
            ('SUE/byte', '@sue_per_byte'),
            ('DQE', '@dqe{0.3f}'),
        ],
        formatters={
            'init_sue': 'printf',
            'final_sue': 'printf',
        },
    )

    # Setup tap tool to jump to description box
    tap = TapTool(
        callback=OpenURL(url="@url", same_tab=True)
    )
    cf.add_tools(hover)
    cf.add_tools(tap)

    # Add DD plots
    dd_plots = []
    for atype in asdp_types:
        fig = figure(
            title='%s DD Components' % atype.upper(),
            plot_width=500, plot_height=500
        )
        x = '%s_x' % atype
        y = '%s_y' % atype
        fig.circle(x, y,
            size=10,
            color={'field': 'asdp_type', 'transform': cmap},
            alpha=0.5,
            hover_color='red',
            hover_alpha=1.0,
            source=source
        )

        fig.xaxis.axis_label = 'DD Component 1'
        fig.yaxis.axis_label = 'DD Component 2'

        fig.add_tools(hover)
        fig.add_tools(tap)

        dd_plots.append(fig)

    # Add cumulative fractional SUE plot
    pf = figure(
        title='Cumulative Fractional SUE',
        plot_width=500, plot_height=500
    )
    for t in asdp_types:
        disp_type = t.upper()
        color_idx = all_types.index(disp_type)
        color = palette[color_idx]

        pf.line(
            x='cumulative_downlink',
            y=('cumulative_sue_%s' % t),
            color=color,
            line_width=3,
            alpha=0.5,
            legend_label=disp_type,
            source=source
        )

    pf.xaxis.axis_label = 'Downlink Data Volume'
    pf.yaxis.axis_label = 'Cumulative Fractional SUE'
    pf.legend.location = 'bottom_right'

    # Arrange plots in grid
    # Use n - 1 columns so cumulative plot appears on a new line
    plotlist = [cf] + dd_plots + [pf]
    plot = gridplot(
        plotlist, ncols=(len(plotlist) - 1)
    )

    # Include complete list of downlink info for ASDPs for jinja template
    template_variables = {
        'downlink_info': downlink_info,
    }

    # Construct HTML file for plots using the extended bokeh template
    html = file_html(plot,
        title="JEWEL Downlink Visualization",
        resources=CDN,
        template=template,
        template_variables=template_variables,
    )


    # Create output directories if they do not already exist
    if not op.exists(outputdir):
        os.makedirs(outputdir)

    detailsdir = op.join(outputdir, DETAILS_DIR)
    if not op.exists(detailsdir):
        os.makedirs(detailsdir)

    details_info = {
        'acme_template': acme_template,
    }

    # Create detail pages for each ASDP
    for i in tqdm(downlink_info, 'Generating Details Pages'):
        create_detail_page(detailsdir, i, details_info)

    # Write main output HTML
    outputfile = op.join(outputdir, HTML_FILE)
    with open(outputfile, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        'sessiondir', help='path to the downlink session directory',
    )
    parser.add_argument(
        'outputdir', help='path to the output dir',
    )

    args = parser.parse_args()
    plot_downlink(**vars(args))


if __name__ == '__main__':
    main()
