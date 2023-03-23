import numpy as np
import pandas as pd
from read_roi import read_roi_file 
import os

'''
https://discuss.datasciencedojo.com/t/shortest-distance-between-points-and-line-segments/1076
Note that there are two implementations on this page, the two give different answers and this one matches an implementation from another web site
p is a single point
P0 and P1 are nx2 2D arrays. P0 contains n starting points, P1 contains n ending points
returns a vector of length n
'''
def pnt2line(P0, P1, p):
    # Calculate the vector T and V
    T = P1 - P0
    V = p - P0
    # Calculate the scalar L and U using dot product
    L = np.sum(T**2, axis=1)
    U = np.sum(T*V, axis=1) / L
    # Clip U to ensure it is within the bounds of [0,1]
    U = np.clip(U, 0, 1)
    # Calculate C using vector T and scalar U
    C = P0 + U[:,np.newaxis]*T
    # Calculate the distance between C and p using norm function
    return np.linalg.norm(C - p, axis=1)


def dist2boundary(cell_file, boundary_roi_file):

    cell_file="M872956_JML_Position8_CD3_img_sizes_coordinates.csv"
    boundary_roi_file="DD_Les_skin_boundary.roi"
    
    boundary = read_roi_file(boundary_roi_file)    
    val = list(boundary.values()) # line_boundary is a dictionary of one item. This line turns it into a list of one item, the item is still a dict
    pts = np.vstack((val[0]["x"], val[0]["y"])).T
    n=len(val[0]["x"])
    
    data = pd.read_csv(cell_file)    

    min_dist=[]
    for index, row in data.iterrows():
        tmp=pnt2line(P0=pts[0:(n-1),:], P1=pts[1:n,:], p=[row["center_x"], row["center_y"]])
        min_dist.append(np.min(tmp))
    
    data = data.join(pd.DataFrame({'min_dist': min_dist}))
    
    filename, file_extension = os.path.splitext(cell_file)
    
    data.to_csv(filename + '_d2b' + file_extension, header=True, index=None, sep=',')
