import os
import pandas as pd
import numpy as np
from tsp import imread
from scipy import ndimage
from tsp.masks import GetCenterCoor 

def IntensityAnalysis(mask_file, image_files, channel=None):
    
    dat = np.load(mask_file, allow_pickle=True).item()
    masks = dat['masks']
    
    filenames=[os.path.splitext(f)[0] for f in image_files]

    centers = GetCenterCoor(masks)
    y_coor=[i[0] for i in centers]
    x_coor=[i[1] for i in centers]

    mask_indices = np.unique(masks)[1:]

    res = [["Cell_"+str(i) for i in mask_indices], x_coor, y_coor]

    for i in range(len(image_files)):
        im = imread(image_files[i])
        if channel is not None: im = im[:,:,channel]
        res.append(ndimage.mean(im, labels=masks, index=mask_indices))
        
    res = pd.DataFrame(res).T
    res.columns = ["name", "x","y"] +filenames
    res.to_csv(os.path.splitext(mask_file)[0] + "_MFI.csv", header=True, index=False, sep=',')



def MeasureIntensity (mask, image, channel=None):
    if channel is not None:
        image = image[:,:,channel-1]
    
    # image_norm = image * (99/255) # normalization for RGB image
    # image_norm = image * (99/65535) # normalization for grayscale image
           
    act_idx = np.unique(mask)
    if(sum(act_idx==0) != 0): act_idx = np.delete(act_idx,0) # select masks only (remove 0)
    intensity = []; intensity_norm_avg_all = []; intensity_norm_avg_pos = []; intensity_norm_total = []
    for j in act_idx :
        mask_pixel = np.where(mask == j) # mask pixels
        pixel_int = []; pixel_norm_int = []
        for k in range(len(mask_pixel[0])):
            pixel_int.append(image[mask_pixel[0][k], mask_pixel[1][k]])
            pixel_norm_int.append(image[mask_pixel[0][k], mask_pixel[1][k]])
            # pixel_norm_int.append(image_norm[mask_pixel[0][k], mask_pixel[1][k]])
        intensity.append(sum(pixel_int)) # total intensities
        intensity_norm_total.append(sum(pixel_norm_int)) # total intensities after normalization
        intensity_norm_avg_all.append(np.mean(pixel_norm_int)) # average intensities of all pixels after normalization
        pixel_norm_int_arr = np.array(pixel_norm_int)

        if(sum(pixel_norm_int_arr != 0)==0):
            intensity_norm_avg_pos.append(0) # average intensities of positive pixels after normalization
        else:
            int_norm_avg_pos = sum(pixel_norm_int_arr[pixel_norm_int_arr != 0]) / sum(pixel_norm_int_arr != 0)
            intensity_norm_avg_pos.append(int_norm_avg_pos) # average intensities of positive pixels after normalization

    return np.around(intensity_norm_total,1), np.around(intensity_norm_avg_all,1), np.around(intensity_norm_avg_pos,1)