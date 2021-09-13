import sys
import os
import os.path      as op
from pathlib        import Path
import glob

import json
import argparse
import logging
import xml.etree.ElementTree    as ET
import numpy                    as np
from scipy.interpolate          import interp1d
import networkx                 as nx


# Modified from LAP_tracker
def export_track_JSON(G, idx, particle_dict, track_dir, track_ext=".json"):
    # list of tracks and their nodes
    ccs = list(nx.connected_components(G))
    if len(ccs) != 1:
        print(f"multiple connected components not expected from {idx}, skipping")
        print(len(G))
        print(len(ccs))
        sys.exit(0)

    # for each connected component
    cc = ccs[0]
    json_dict = {
        'Times': [],
        'Particles_Position': [],
        'Particles_Estimated_Position': [],
        'Particles_Size': [],
        'Particles_Bbox': [],
        'Particles_Max_Intensity': [],
        'Track_ID': idx,
        'classification': None
    }

    # sort track by timestamp
    cc_sorted = sorted(cc, key = lambda x: x[2])
    cc_coords = [[c[0], c[1]] for c in cc_sorted]
    cc_times = [int(c[2]) for c in cc_sorted]

    # function for interpolation
    interp_func = interp1d(cc_times, cc_coords, kind='linear', axis=0)

    # for each timestep in timerange
    for t in range(cc_times[0], cc_times[-1]+1):
        json_dict['Times'].append(t)

        if t in cc_times:
            # particle exists, no interpolation
            # get particle object
            particle = particle_dict[cc_sorted[cc_times.index(t)]]
            json_dict['Particles_Position'].append(particle['pos'])
            json_dict['Particles_Estimated_Position'].append(particle['pos'])
            json_dict['Particles_Size'].append(particle['size'])
            json_dict['Particles_Bbox'].append(particle['bbox'])
            json_dict['Particles_Max_Intensity'].append(particle['max_intensity'])

        else:
            # particle DNE, interpolate
            json_dict['Particles_Estimated_Position'].append(interp_func(t).tolist())
            json_dict['Particles_Position'].append(None)
            json_dict['Particles_Size'].append(None)
            json_dict['Particles_Bbox'].append(None)
            json_dict['Particles_Max_Intensity'].append(None)

    # save dictionary to JSON
    json_fpath = op.join(track_dir, f'{idx:05}{track_ext}')
    with open(json_fpath, 'w') as f:
        json.dump(json_dict, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('xmlpath',        type=str,
                                            help="Filepath to the XML file to be converted")

    parser.add_argument('--json_outdir',    type=str,
                                            default="outdir/",
                                            help="JSON output directory path")

    args = parser.parse_args()

    if not op.isdir(args.json_outdir):
        os.mkdir(args.json_outdir)

    
    ############
    # Read XML #
    ############

    tree = ET.parse(args.xmlpath)
    root = tree.getroot()
    model = root.find('Model')


    # Read Spots
    ############

    particle_dict = {}
    particleID_map = {}
    spots = model.find('AllSpots')
    for frame in spots.iter('SpotsInFrame'):
        frame_n = int(frame.get('frame'))
        for spot in frame.iter('Spot'):
            particle_tup = (float(spot.get('POSITION_X')), float(spot.get('POSITION_Y')), frame_n)
            particle_dict[particle_tup] = {
                'pos': particle_tup[:2],
                'size': 1,
                'bbox': [
                    [float(spot.get('POSITION_X')) - float(spot.get('ESTIMATED_DIAMETER')) / 2,
                     float(spot.get('POSITION_Y')) - float(spot.get('ESTIMATED_DIAMETER')) / 2],
                    [float(spot.get('ESTIMATED_DIAMETER')),
                     float(spot.get('ESTIMATED_DIAMETER'))]
                ],
                'max_intensity': spot.get('MAX_INTENSITY')
            }
            particleID_map[spot.get('ID')] = particle_tup


    # Read Edges
    ############

    tracks = model.find('AllTracks')
    for track in tracks.iter('Track'):
        G = nx.Graph()
        track_ID = int(track.get('TRACK_ID'))
        counter = 0
        for edge in track.iter('Edge'):
            G.add_edge(particleID_map[edge.get('SPOT_SOURCE_ID')], 
                       particleID_map[edge.get('SPOT_TARGET_ID')])
            counter += 1
        export_track_JSON(G, track_ID, particle_dict, args.json_outdir)

    print(f"Converted files saved to {args.json_outdir}")