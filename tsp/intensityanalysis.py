import os
import pandas as pd
import numpy as np
from tsp import imread
from scipy import ndimage
from tsp.masks import GetCenterCoor 

def IntensityAnalysis(mask_file, image_files, channel=None):
    
    masks = imread(mask_file)

    centers = GetCenterCoor(masks)
    y_coor=[i[0] for i in centers]
    x_coor=[i[1] for i in centers]

    mask_indices = np.unique(masks)[1:]
    cell_name = ["Cell_"+str(i) for i in mask_indices]

    res = [cell_name, x_coor, y_coor]

    for i in range(len(image_files)):
        im = imread(image_files[i])
        if channel is not None: im = im[:,:,channel]
        res.append(ndimage.mean(im, labels=masks, index=mask_indices))
        
    res = pd.DataFrame(res).T
    res.columns = ["name", "x","y"] + [os.path.splitext(f)[0] for f in image_files]
    res.to_csv(os.path.splitext(mask_file)[0] + "_MFI.csv", header=True, index=False, sep=',')


