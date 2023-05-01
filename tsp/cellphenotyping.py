import os, math, sys
import pandas as pd
import numpy as np
from tsp.masks import GetCenterCoor #, PlotMask_outline, PlotMask_center
from tsp import imread
import skimage.io
from skimage import img_as_ubyte, img_as_uint
from cellpose import utils
import timeit
from scipy import ndimage


def StainingAnalysis(files, marker_names, positives, cutoffs, channels, methods, save_plot):
    
    plus_minus = ['+' if positives[l] else '-' for l in range(len(marker_names))]
    filenames=[os.path.splitext(f)[0] for f in files]
    staged_output_file_names = [filenames[0]]
    tmp =  [marker_names[i] + plus_minus[i] + str(cutoffs[i]) for i in range(len(marker_names))]
    staged_output_file_names = staged_output_file_names + [filenames[0]+"_"+"".join(tmp[:(i+1)]) for i in range(len(marker_names))]
    output_file_name = staged_output_file_names[-1]
        
    pos_rates = []; num_cells = []; mask_idxes = []; masks = []
        
    # datA = np.load(os.path.splitext(files[0])[0] + '_seg.npy', allow_pickle=True).item()
    # maskA = datA['masks']
    maskA = imread(os.path.splitext(files[0])[0] + '_masks_id.png')
    masks.append(maskA)
    num_cells.append(maskA.max())
    
    n_markers = len(files)-1 # not counting ref marker

    start_time = timeit.default_timer()
    
    for i in range(n_markers):
        positive=positives[i]
        cutoff=cutoffs[i]
        method=methods[i]
        channel=channels[i] if channels is not None else None
            
        # Method (Positivity or Intensity) #
        if(method == 'Mask'):
            # datB = np.load(os.path.splitext(files[i+1])[0] + '_seg.npy', allow_pickle=True).item()
            # maskB = datB['masks']
            maskB = imread(os.path.splitext(files[i+1])[0] + '_masks_id.png')
        elif(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
            image_comp = imread(files[i+1])
        else:
            sys.exit("method incorrectly specified")
        
        # Double staining #
        if(method == 'Mask'):
            pos_rate, num_double_cell, double_mask_idx = DoubleStain(maskA=maskA, maskB=maskB, positive=positive, cutoff=cutoff, channel=channel, method=method)
        elif(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
            pos_rate, num_double_cell, double_mask_idx = DoubleStain(maskA=maskA, maskB=image_comp, positive=positive, cutoff=cutoff, channel=channel, method=method)

        # for the last file, examine a series of cutoffs. this step does not take too much time
        if(i == n_markers-1):
            if(method == 'Mask'):
                cutoff_all = list(np.around(np.linspace(start=0, stop=1, num=11),1)) # [0,0.1,...,0.9,1]                
            elif(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
                cutoff_all = list(np.around(np.quantile(pos_rate, np.linspace(start=0, stop=1, num=11)),1)) # quantile
                #cutoff_all = [0.0,0.5,0.7,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,4.9,5.0,5.5,6.0,7.0,8.0,9.0,10.0] # for severity analysis
            
            num_double_cell_cutoff = [] # number of double stained cell over all cutoffs
            for k in cutoff_all: 
                num_cell_temp = []
                for j in np.arange(0,len(pos_rate)) :
                    if positive: 
                        num_cell_temp.append(pos_rate[j] >= k)
                    else:
                        num_cell_temp.append(pos_rate[j] <= k)
                num_double_cell_cutoff.append(sum(num_cell_temp))
            ncell_res_temp = pd.DataFrame(list(zip(cutoff_all, num_double_cell_cutoff)))
            ncell_res_temp.columns = ["Cutoff", "Cell_count"]
            ncell_res_temp.to_csv(output_file_name + "_counts_lastcutoff.txt", header=True, index=None, sep=',')
        
        pos_rates.append(pos_rate)
        num_cells.append(num_double_cell)
        mask_idxes.append(double_mask_idx)
        maskA = GetMaskCutoff(mask=maskA, act_mask_idx=double_mask_idx)
        masks.append(maskA)
        
        print(f"time spent {timeit.default_timer() - start_time}"); start_time = timeit.default_timer()

    
    # save masks to a csv file
    for i in range(n_markers):

        tmp=np.unique(masks[i+1], return_counts=True)
        sizes = tmp[1][1:]#.tolist()    # keep it as an array     
        ncell=len(sizes)
    
        centers=GetCenterCoor(masks[i+1])
        y_coor, x_coor = zip(*centers)
        # turn tuples into arrays to use as.type later
        y_coor=np.array(y_coor); x_coor=np.array(x_coor)
    
        ## Save a csv file of mask info. One row per mask, columns include size, center_x, center_y
        mask_info = pd.DataFrame({
            "center_x": x_coor, 
            "center_y": y_coor,
            "size": sizes
        })
        mask_info.index = [f"Cell_{i}" for i in range(1,ncell+1)]
        mask_info=mask_info.round().astype(int)
        mask_info.to_csv(output_file_name + "_masks.csv", header=True, index=True, sep=',')
    

    print(f"time spent {timeit.default_timer() - start_time}"); start_time = timeit.default_timer()

    # save counts
    filenames_save = [files[0]] # first filename
    for i in range(len(files)-1):
        if positives[i]: 
            filenames_save.append("+" + files[i+1])
        else:
            filenames_save.append("-" + files[i+1])
    ncell_res = pd.DataFrame(list(zip(filenames_save, num_cells)))
    ncell_res.columns = ["File_name", "Cell_count"]
    ncell_res.to_csv(output_file_name + "_counts_multistain.txt", header=True, index=None, sep=',')

    print(f"time spent {timeit.default_timer() - start_time}"); start_time = timeit.default_timer()
    
    for i in range(len(masks)):
        # PlotMask_outline(mask=masks[i], img=files[i], savefilename=staged_output_file_names[i] + '_outline.png', color=mask_color)
        skimage.io.imsave(staged_output_file_names[i] + '_masks.png', img_as_ubyte(utils.masks_to_outlines(masks[i])))
        skimage.io.imsave(staged_output_file_names[i] + '_masks_id.png', img_as_uint(masks[i]), check_contrast=False)

        # PlotMask_outline(mask=masks[i], img=files[i], savefilename=staged_output_file_names[i] + '_fill.png',    color=[255,255,255], fill=True)
        if(save_plot): skimage.io.imsave(staged_output_file_names[i] + '_masks_fill.png', img_as_ubyte(masks[i]!=0))

        # PlotMask_center (mask=masks[i], img=files[i], savefilename=staged_output_file_names[i] + '_point.png',   color='r')
    
    # for i in range(len(files)-1):
    #     np.savez(file=output_file_name + '_seg', img=image_base, masks=masks[i+1])
            
    print(f"time spent {timeit.default_timer() - start_time}"); start_time = timeit.default_timer()


# Utilites for double staining analysis
def DoubleStain(maskA, maskB, positive, cutoff, channel, method):
    
    # Pre-processing #
    if(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
        if maskB.ndim==3:
            if channel is not None: 
                maskB = maskB[:,:,channel]
            else:
                sys.exit("--l is required when intensity is used and image is RGB")

        maskB = maskB * (99/255) # normalization 255 for 8 bit, 65535 for 16 bit grayscale
    
    # Double staining #
    
    tab = np.histogram2d(maskA.flatten(), maskB.flatten(), bins=[np.append(np.unique(maskA), np.inf), np.append(np.unique(maskB), np.inf)])[0]
    size, mask_indices = np.histogram(maskA, bins=np.append(np.unique(maskA), np.inf))
    mask_indices=mask_indices[1:-1]
    res = tab[1:,1:].max(axis=1) / size[1:]
    
    if(method == 'Intensity_total'):
        res = ndimage.sum(maskB, labels=maskA, index=mask_indices)
    if(method == 'Intensity_avg_all'):
        res  = ndimage.mean(maskB, labels=maskA, index=mask_indices)
                
    # if(method == 'Intensity_avg_pos' or method == 'Intensity_avg_all' or method == 'Intensity_total'):
    #     act_idx = np.unique(maskA)[1:]
    #     res=[]
    #     for i in act_idx:
    #         intensity_temp = []
    #         cell = np.where(maskA == i)
    #         for j in range(len(cell[0])) :
    #             temp = maskB[cell[0][j], cell[1][j]]
    #             intensity_temp.append(temp)
    #         if(method == 'Intensity_total'):
    #             res.append(sum(intensity_temp)) # total intensity
    #         if(method == 'Intensity_avg_pos'):
    #             intensity_temp_arr = np.array(intensity_temp)
    #             int_norm_avg_pos = sum(intensity_temp_arr[intensity_temp_arr != 0]) / sum(intensity_temp_arr != 0)
    #             if(math.isnan(int_norm_avg_pos)):
    #                 res.append(0) # average intensities of positive pixels after normalization
    #             else:
    #                 res.append(int_norm_avg_pos) # average intensities of positive pixels after normalization
    #         if(method == 'Intensity_avg_all'):
    #             res.append(np.mean(intensity_temp)) # average intensities of all pixels after normalization
    
    if positive: 
        double_mask_idx = mask_indices[res >= cutoff]
    else:
        double_mask_idx = mask_indices[res <= cutoff]

    return res, len(double_mask_idx), double_mask_idx



def GetMaskCutoff(mask, act_mask_idx):
    act_idx = act_mask_idx
    total_idx = np.unique(mask)
    inact_idx = np.array(list(set(total_idx) - set(act_idx))) #inact_idx = np.delete(total_idx, mask_idx)
    mask_cutoff = mask.copy()
    for i in np.arange(0,len(inact_idx)) :
        mask_cutoff[mask_cutoff==inact_idx[i]] = 0
    return mask_cutoff

