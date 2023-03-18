import os, math
import pandas as pd
import numpy as np
from tsp.masks import GetCenterCoor
from tsp import imread


def IntensityAnalysis(files, channels):
    filenames=[os.path.splitext(f)[0] for f in files]
    image_base = imread(files[0])
    mask_path = filenames[0] + '_seg.npy'
    dat = np.load(mask_path, allow_pickle=True).item()
    mask = dat['masks']
    outlines = GetCenterCoor(mask)
    
    intensity_total = []
    for i in range(len(files)):
        if(i == 0):
            res = MeasureIntensity(mask=mask, image=image_base, channels=channels)
        else:
            image_comp = imread(files[i])
            res = MeasureIntensity(mask=mask, image=image_comp, channels=channels)
        intensity_total.append(res.intensity_norm_avg_all); 
        intensity_total.append(res.intensity_norm_avg_pos); 
        intensity_total.append(res.intensity_norm_total)
    intensity_total.append(list(outlines))
    intensity_res = pd.DataFrame(intensity_total).T
    colnames = []
    for i in range(len(filenames)):
        temp = [filenames[i] + "_intensity_avg_all", filenames[i] + "_intensity_avg_pos", filenames[i] + "_intensity_total"]
        for j in range(3):
           colnames.append(temp[j])
    colnames.append("xy_coordinate")
    intensity_res.columns = colnames
    cellnames = []
    for i in range(intensity_res.shape[0]): 
        cellnames.append("Cell_" + str(i+1))
    intensity_res.index = cellnames
    intensity_res.to_csv(filenames[0] + "_intensity.txt", header=True, index=True, sep=',')



class MeasureIntensity:
    def __init__(self, mask, image, channels):
        if(channels != [0,0]): image = image[:,:,(channels[0]-1)]
        
        if(channels != [0,0]): image_norm = image * (99/255) # normalization for RGB image
        if(channels == [0,0]): image_norm = image * (99/65535) # normalization for grayscale image
               
        act_idx = np.unique(mask)
        if(sum(act_idx==0) != 0): act_idx = np.delete(act_idx,0) # select masks only (remove 0)
        intensity = []; intensity_norm_avg_all = []; intensity_norm_avg_pos = []; intensity_norm_total = []
        for j in act_idx :
            mask_pixel = np.where(mask == j) # mask pixels
            pixel_int = []; pixel_norm_int = []
            for k in range(len(mask_pixel[0])):
                pixel_int.append(image[mask_pixel[0][k], mask_pixel[1][k]])
                pixel_norm_int.append(image_norm[mask_pixel[0][k], mask_pixel[1][k]])
            intensity.append(sum(pixel_int)) # total intensities
            intensity_norm_total.append(sum(pixel_norm_int)) # total intensities after normalization
            intensity_norm_avg_all.append(np.mean(pixel_norm_int)) # average intensities of all pixels after normalization
            pixel_norm_int_arr = np.array(pixel_norm_int)
            int_norm_avg_pos = sum(pixel_norm_int_arr[pixel_norm_int_arr != 0]) / sum(pixel_norm_int_arr != 0)
            if(math.isnan(int_norm_avg_pos)):
                intensity_norm_avg_pos.append(0) # average intensities of positive pixels after normalization
            else:
                intensity_norm_avg_pos.append(int_norm_avg_pos) # average intensities of positive pixels after normalization
        self.intensity = np.around(intensity,1)
        self.intensity_norm_total = np.around(intensity_norm_total,1)
        self.intensity_norm_avg_all = np.around(intensity_norm_avg_all,1)
        self.intensity_norm_avg_pos = np.around(intensity_norm_avg_pos,1)

