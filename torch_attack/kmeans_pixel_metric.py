from pyclustering.utils.metric import type_metric, distance_metric

from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import numpy as np
# Load list of points for cluster analysis.
def user_function(p1,p2):
#    delta_x = pow(p1[0]-p2[0],2)
#    delta_y = pow(p1[1]-p2[1],2)
    delta_x = abs(p1[0]-p2[0])
    delta_y = abs(p1[1]-p2[1])
    return np.maximum(delta_x,delta_y)

def get_cluster_custom(sample,visualize=False,class_num=3):
    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

    # create K-Means algorithm with specific distance metric
    initial_centers = kmeans_plusplus_initializer(sample, class_num).initialize()
    kmeans_instance = kmeans(sample, initial_centers, metric=metric)

    # run cluster analysis and obtain results
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    cls_encoded = []
    for cls_idx in clusters:
        cls_encoded.append(sample[cls_idx])
    final_centers = kmeans_instance.get_centers()
    for i in range(len(final_centers)):
        final_centers[i] = np.round(np.array(final_centers[i]))
    loss = kmeans_instance.get_total_wce()
    # Visualize obtained results
    if visualize:
        kmeans_visualizer.show_clusters(sample, clusters, final_centers)
    return cls_encoded, final_centers, loss
from pyclustering.utils.metric import type_metric, distance_metric

from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import numpy as np
# Load list of points for cluster analysis.
def user_function(p1,p2):
#    delta_x = pow(p1[0]-p2[0],2)
#    delta_y = pow(p1[1]-p2[1],2)
    delta_x = abs(p1[0]-p2[0])
    delta_y = abs(p1[1]-p2[1])
    return np.maximum(delta_x,delta_y)

def get_cluster_custom(sample,visualize=False,class_num=3):
    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)

    # create K-Means algorithm with specific distance metric
    initial_centers = kmeans_plusplus_initializer(sample, class_num).initialize()
    kmeans_instance = kmeans(sample, initial_centers, metric=metric)

    # run cluster analysis and obtain results
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    cls_encoded = []
    for cls_idx in clusters:
        cls_encoded.append(sample[cls_idx])
    final_centers = kmeans_instance.get_centers()
    for i in range(len(final_centers)):
        final_centers[i] = np.round(np.array(final_centers[i]))
    loss = kmeans_instance.get_total_wce()
    # Visualize obtained results
    if visualize:
        kmeans_visualizer.show_clusters(sample, clusters, final_centers)
    return cls_encoded, final_centers, loss
