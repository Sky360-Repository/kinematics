# utility functions
from scipy.spatial import distance
import math
import numpy as np

def center_of_mass(bbox):
    height=bbox[2]
    width=bbox[3]
    x = bbox[0] + height/2
    y = bbox[1] + width/2
    return (x,y)

def distance_vector(tuplea,tupleb):
    from scipy.spatial import distance
    # calculate euclidean distances [px] of a bounding box between two images
    dist = distance.euclidean(tupleb,tuplea)
    x = tupleb[0] - tuplea[0]
    y = tupleb[1] - tuplea[1]
    vector = (x,y)
    return vector,dist

def direction_change(a,b):
    # calculate angle between two subsequent direction vectors calculated from bounding box[i] and [i-1]
    magn_a = math.sqrt(a[0]**2 + a[1]**2) # magnitude of vector a
    magn_b = math.sqrt(b[0]**2 + b[1]**2) # magnitude of vector b
    cos_theta = np.dot(a,b)/(magn_a * magn_b) # cos of angle between vector a and b
    theta = np.arccos(cos_theta) * 180/math.pi # angle in degree
    return theta

def clean_nans(list_a):
    list_cleaned = []
    for i in range(0,len(list_a)):
        if str(list_a[i]) == 'nan':
            list_cleaned.append(0)
        else:
            list_cleaned.append(list_a[i])
    return list_cleaned
