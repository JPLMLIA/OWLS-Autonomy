import os
import os.path      as op
from pathlib        import Path
import glob

import json
import xml.etree.ElementTree as ET
import argparse
import logging
import numpy as np

def stringify(d):
    converted = {}
    for k in d:
        converted[k] = str(d[k])
    return converted

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('track_dir',        type=str,
                                            help="Directory to track JSONs to be converted")

    parser.add_argument('--xml_outpath',    type=str,
                                            default="converted.xml",
                                            help="XML output filepath")

    args = parser.parse_args()

    track_dir = args.track_dir
    if not op.isdir(track_dir):
        logging.error(f"{track_dir} does not exist.")

    track_jsons = sorted(glob.glob(op.join(track_dir, "*.json")))
    if len(track_jsons) == 0:
        logging.error(f"{track_dir} has no JSONs.")

    #############
    # READ JSON #
    #############

    # This stores spots on a frame-by-frame basis
    spot_storage = {}
    # This stores spots on a track-by-track basis
    track_storage = {}
    global_spot_id = 0
    for track_id, track_fp in enumerate(track_jsons):
        with open(track_fp, 'r') as f:
            track_data = json.load(f)

            prev = None
            for t, pos, bbox, size, intensity in zip(track_data['Times'],
                                                     track_data['Particles_Estimated_Position'],
                                                     track_data['Particles_Bbox'],
                                                     track_data['Particles_Size'],
                                                     track_data['Particles_Max_Intensity']):
                
                # Initialize list if it doesn't exist already
                if int(t) not in spot_storage:
                    spot_storage[int(t)] = []
                
                # Estimate a radius from bounding box'
                if bbox is not None:
                    radius = np.mean(np.array(bbox[1])/2)
                else:
                    radius = 0
                
                # Add spot to spot storage
                if size is not None:
                    spot = {
                        'ID': global_spot_id,
                        'name': f"ID{global_spot_id}",
                        'POSITION_X': pos[0],
                        'POSITION_Y': pos[1],
                        'POSITION_Z': 0,
                        'POSITION_T': float(t),
                        'FRAME': int(t),
                        'MAX_INTENSITY': intensity,
                        'SIZE': size,
                        'VISIBILITY': 1,
                        'RADIUS': radius
                    }
                else:
                    spot = {
                        'ID': global_spot_id,
                        'name': f"ID{global_spot_id}",
                        'POSITION_X': pos[0],
                        'POSITION_Y': pos[1],
                        'POSITION_Z': 0,
                        'POSITION_T': float(t),
                        'FRAME': int(t),
                        'MAX_INTENSITY': 0,
                        'SIZE': 0,
                        'VISIBILITY': 1,
                        'RADIUS': radius
                    }


                spot_storage[int(t)].append(spot)

                # Add spot to track storage
                if track_id not in track_storage:
                    track_storage[track_id] = []
                
                track_storage[track_id].append(spot)

                # Increment global spot ID
                global_spot_id += 1

    # This stores edges for each track
    edge_storage = {}
    for track_id in track_storage:
        track_spots = track_storage[track_id]
        edge_storage[track_id] = []
        for i in range(1, len(track_spots)):
            edge_storage[track_id].append({
                'SPOT_SOURCE_ID': track_spots[i-1]['ID'],
                'SPOT_TARGET_ID': track_spots[i]['ID'],
                'LINK_COST': np.linalg.norm(np.array([track_spots[i-1]['POSITION_X'], track_spots[i-1]['POSITION_Y']]) -
                                            np.array([track_spots[i]['POSITION_X'], track_spots[i]['POSITION_Y']])),
                'EDGE_TIME': np.mean([track_spots[i-1]['POSITION_T'], track_spots[i]['POSITION_T']]),
                'EDGE_X_LOCATION': np.mean([track_spots[i-1]['POSITION_X'], track_spots[i]['POSITION_X']]),
                'EDGE_Y_LOCATION': np.mean([track_spots[i-1]['POSITION_Y'], track_spots[i]['POSITION_Y']]),
                'EDGE_Z_LOCATION': np.mean([track_spots[i-1]['POSITION_Z'], track_spots[i]['POSITION_Z']]),
                'VELOCITY': np.linalg.norm(np.array([track_spots[i-1]['POSITION_X'], track_spots[i-1]['POSITION_Y']]) -
                                           np.array([track_spots[i]['POSITION_X'], track_spots[i]['POSITION_Y']])),
                'DISPLACEMENT': np.linalg.norm(np.array([track_spots[i-1]['POSITION_X'], track_spots[i-1]['POSITION_Y']]) -
                                               np.array([track_spots[i]['POSITION_X'], track_spots[i]['POSITION_Y']])),
            })


    # This stores track features
    trackfeat_storage = {}
    for track_id in edge_storage:
        trackfeat_storage[track_id] = {
            'NUMBER_SPOTS': len(edge_storage[track_id])+1,
            'TRACK_DURATION': len(edge_storage[track_id]),
            'TRACK_START': min([x['POSITION_T'] for x in track_storage[track_id]]),
            'TRACK_STOP': max([x['POSITION_T'] for x in track_storage[track_id]]),
            'TRACK_DISPLACEMENT': np.linalg.norm(np.array([track_storage[track_id][0]['POSITION_X'], track_storage[track_id][0]['POSITION_Y']]) - 
                                                 np.array([track_storage[track_id][-1]['POSITION_X'], track_storage[track_id][-1]['POSITION_X']])),
            'TRACK_INDEX': track_id,
            'TRACK_ID': track_id,
            'name': f"Track_{track_id}",
            'TRACK_X_LOCATION': np.mean([x['POSITION_X'] for x in track_storage[track_id]]),
            'TRACK_Y_LOCATION': np.mean([x['POSITION_Y'] for x in track_storage[track_id]]),
            'TRACK_Z_LOCATION': np.mean([x['POSITION_Z'] for x in track_storage[track_id]]),
        }


    #########################
    # TRACKMATE XML BUILDER #
    #########################


    # Top-Level Hierarchy
    #####################

    data = ET.Element("TrackMate", {"version": "6.0.2"})
    log = ET.SubElement(data, "Log")
    log.text = "This TrackMate XML file was generated by the JSON-to-XML converter \n\
of the OWLS-Autonomy package. Some features may be missing."
    model = ET.SubElement(data, "Model", {"spatialunits": "pixel", "timeunits": "frame"})


    # Feature Declarations
    ######################

    featdec = ET.SubElement(model, "FeatureDeclarations")

    # Spot Features
    spotfeat = ET.SubElement(featdec, "SpotFeatures")
    spot_features = [
        {"feature": "POSITION_X",
         "name": "X",
         "shortname": "X",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "POSITION_Y",
         "name": "Y",
         "shortname": "Y",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "POSITION_Z",
         "name": "Z",
         "shortname": "Z",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "POSITION_T",
         "name": "T",
         "shortname": "T",
         "dimension": "TIME",
         "isint": "false"},
        {"feature": "FRAME",
         "name": "Frame",
         "shortname": "Frame",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "MAX_INTENSITY",
         "name": "Maximal intensity",
         "shortname": "Max",
         "dimension": "INTENSITY",
         "isint": "false"},
        {"feature": "SIZE",
         "name": "Number of pixels in particle",
         "shortname": "Size",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "VISIBILITY",
         "name": "Visibility",
         "shortname": "Visibility",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "RADIUS",
         "name": "radius",
         "shortname": "R",
         "dimension": "LENGTH",
         "isint": "false"}
    ]

    for sf in spot_features:
        ET.SubElement(spotfeat, "Feature", sf)
    
    # Edge Features
    edgefeat = ET.SubElement(featdec, "EdgeFeatures")
    edge_features = [
        {"feature": "SPOT_SOURCE_ID",
         "name": "Source spot ID",
         "shortname": "Source ID",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "SPOT_TARGET_ID",
         "name": "Target spot ID",
         "shortname": "Target ID",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "LINK_COST",
         "name": "Link cost",
         "shortname": "Cost",
         "dimension": "NONE",
         "isint": "false"},
        {"feature": "EDGE_TIME",
         "name": "Time (mean)",
         "shortname": "T",
         "dimension": "TIME",
         "isint": "false"},
        {"feature": "EDGE_X_LOCATION",
         "name": "X Location (mean)",
         "shortname": "X",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "EDGE_Y_LOCATION",
         "name": "Y Location (mean)",
         "shortname": "Y",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "EDGE_Z_LOCATION",
         "name": "Z Location (mean)",
         "shortname": "Z",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "VELOCITY",
         "name": "Velocity",
         "shortname": "V",
         "dimension": "VELOCITY",
         "isint": "false"},
        {"feature": "DISPLACEMENT",
         "name": "Displacement",
         "shortname": "D",
         "dimension": "LENGTH",
         "isint": "false"}
    ]

    for ef in edge_features:
        ET.SubElement(edgefeat, "Feature", ef)

    # Track Features
    trackfeat = ET.SubElement(featdec, "TrackFeatures")

    track_features = [
        {"feature": "NUMBER_SPOTS",
         "name": "Number of spots in track",
         "shortname": "N spots",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "TRACK_DURATION",
         "name": "Duration of track",
         "shortname": "Duration",
         "dimension": "TIME",
         "isint": "false"},
        {"feature": "TRACK_START",
         "name": "Track start",
         "shortname": "T start",
         "dimension": "TIME",
         "isint": "false"},
        {"feature": "TRACK_STOP",
         "name": "Track stop",
         "shortname": "T stop",
         "dimension": "TIME",
         "isint": "false"},
        {"feature": "TRACK_DISPLACEMENT",
         "name": "Track displacement",
         "shortname": "Displacement",
         "dimension": "LENGTH",
         "isint": "false"},
        {"feature": "TRACK_INDEX",
         "name": "Track index",
         "shortname": "Index",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "TRACK_ID",
         "name": "Track ID",
         "shortname": "ID",
         "dimension": "NONE",
         "isint": "true"},
        {"feature": "TRACK_X_LOCATION",
         "name": "X Location (mean)",
         "shortname": "X",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "TRACK_Y_LOCATION",
         "name": "Y Location (mean)",
         "shortname": "Y",
         "dimension": "POSITION",
         "isint": "false"},
        {"feature": "TRACK_Z_LOCATION",
         "name": "Z Location (mean)",
         "shortname": "Z",
         "dimension": "POSITION",
         "isint": "false"}
    ]
    
    for tf in track_features:
        ET.SubElement(trackfeat, "Feature", tf)

    
    # Spots
    #######

    allspots = ET.SubElement(model, "AllSpots", {"nspots": str(np.sum([len(spot_storage[x]) for x in spot_storage]))})
    for frame in spot_storage:
        sif = ET.SubElement(allspots, "SpotsInFrame", {"frame": str(frame)})
        for spot in spot_storage[frame]:
            ET.SubElement(sif, "Spot", stringify(spot))

    # Tracks
    ########

    alltracks = ET.SubElement(model, "AllTracks")
    for track_id in trackfeat_storage:
        curr_feats = trackfeat_storage[track_id]
        curr_track = ET.SubElement(alltracks, "Track", stringify(curr_feats))

        for edge in edge_storage[track_id]:
            ET.SubElement(curr_track, "Edge", stringify(edge))
    
    # Filtered Tracks
    #################
    filtracks = ET.SubElement(model, "FilteredTracks")
    for track_id in trackfeat_storage:
        ET.SubElement(filtracks, "TrackID", {"TRACK_ID": str(track_id)})

    
    ET.ElementTree(data).write(args.xml_outpath, encoding='UTF-8', xml_declaration=True) 
    print(f"Converted file saved to {args.xml_outpath}")