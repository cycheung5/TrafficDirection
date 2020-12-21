

import random
from sklearn.cluster import KMeans
import numpy as np
import os
import argparse


class TrackletClustering(object):
  

    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.tracklet_list = []
        self.cluster_list = []

    def add_tracklet(self, tracklet):
        "Add a new tracklet into the database"
        self.tracklet_list.append(tracklet)

        
    def build_clustering_model(self):
        "Perform clustering algorithm"
        arr_length = len(self.tracklet_list)
        arr = np.empty([arr_length, 4])
        counter = 0
        for tracklet in self.tracklet_list:
            v_det = tracklet["tracks"]
            first_box = v_det[0][1:]
            last_box = v_det[-1][1:]
            # Get midpoint 
            first_midpoint_x = (first_box[0] + first_box[2]) // 2
            first_midpoint_y = (first_box[1] + first_box[3]) // 2
            last_midpoint_x = (last_box[0] + last_box[2]) // 2
            last_midpoint_y = (last_box[1] + last_box[3]) // 2
            features = [first_midpoint_x, first_midpoint_y, last_midpoint_x, last_midpoint_y]
            #features = [last_midpoint_x - first_midpoint_x, last_midpoint_y - first_midpoint_y]
            arr[counter] = features
            counter = counter + 1
        # Now need to put the sklearn k means
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(arr)
        self.cluster_list = kmeans.labels_
        

    def get_cluster_id(self, tracklet):
        """
        Assign the cluster ID for a tracklet. This funciton must return a non-negative integer <= num_cluster
        It is possible to return value 0, but it is reserved for special category of abnormal behavior (for Question 2.3)
        """
        ind = self.tracklet_list.index(tracklet)
        v_cluster_id = int(self.cluster_list[ind]) + 1
        assert v_cluster_id >=0, "direction_id cannot be negative"
        return v_cluster_id
        #print(kmeans.labels_)

        #return random.randint(0, self.num_cluster)
