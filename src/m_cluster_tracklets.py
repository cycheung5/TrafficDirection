#!/usr/bin/env python


import argparse
import json
from hw5_tracklet_clutering import TrackletClustering
from tqdm import tqdm
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cluster vehicle tracklets")
    parser.add_argument("-i", "--in_tracket_file", type=str, required=True, help="/path/to/input/json/file/of/tracking/result")
    parser.add_argument("-o", "--out_tracklet_file", type=str, help="/path/to/output/json/file/of/tracking/result/with/clustering/info")
    parser.add_argument("-n", "--num_cluster", type=int, default=3)
    args = parser.parse_args()

    with open(args.in_tracket_file) as f:
        vehicle_data = json.load(f)

    clust_obj = TrackletClustering(args.num_cluster)

    # Gather the data
    print("First pass over data to gather all tracklets")
    for v_id in tqdm(vehicle_data):
        v_tracklet = vehicle_data[v_id]
        clust_obj.add_tracklet(v_tracklet)
        
    # build clustering model
    clust_obj.build_clustering_model()

    # perform assignment
    print("Second pass over data to assign tracklets to clusters")
    for v_id in tqdm(vehicle_data):
        v_tracklet = vehicle_data[v_id]
        v_tracklet["direction_id"] = clust_obj.get_cluster_id(v_tracklet) # obtain cluster id and update the data file

    if args.out_tracklet_file is None:
        dir_path = os.path.dirname(args.in_tracket_file)
        out_tracket_file = "{}/out_{}".format(dir_path, os.path.basename(args.in_tracket_file))
    else:
        out_tracket_file = args.out_tracklet_file

    # save the results to output data file
    with open(out_tracket_file, "w") as f:
        json.dump(vehicle_data, f, indent=4)