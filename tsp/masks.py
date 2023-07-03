import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_roi import read_roi_file 
from PIL import Image, ImageDraw
from scipy import ndimage
from cellpose import utils, io
from tsp import imread, imsave
import skimage.io
from skimage import img_as_ubyte, img_as_uint
from tsp.AP import compute_iou, tp_fp_fn


# copied from cellpose
# Masks to outlines
def masks_to_outlines(masks):
    """ get outlines of masks as a 0-1 array 
    
    Parameters
    ----------------

    masks: int, 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    outlines: 2D or 3D array 
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    outlines = np.zeros(masks.shape, bool)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = ndimage.find_objects(masks.astype(int))
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
                vr, vc = pvr + sr.start, pvc + sc.start 
                outlines[vr, vc] = 1
        return outlines


# From .roi files to masks file
def roifiles2mask(files, width, height, saveas, save_masks_only=False):
    print("number of roi files: "+str(len(files)))
    masks = Image.new('I', (width, height), 0)
    for idx in range(len(files)):
        mask_temp = read_roi_file(files[idx])
        filename = files[idx].split(os.sep)[-1][:-4]
        mask_temp = mask_temp[filename]
        if mask_temp['type'] == 'rectangle':
            x = [mask_temp['left'], mask_temp['left']+mask_temp['width'], mask_temp['left']+mask_temp['width'], mask_temp['left']]
            y = [mask_temp['top'],  mask_temp['top'], mask_temp['top']+mask_temp['height'], mask_temp['top']+mask_temp['height']]
        else:
            x = mask_temp['x']
            y = mask_temp['y']
            
        polygon = []
        for i in range(len(x)):
            polygon.append((x[i], y[i]))
        
        ImageDraw.Draw(masks).polygon(polygon, outline=idx+1, fill=idx+1)
            
    masks = np.array(masks, dtype=np.uint16) # resulting masks
    outlines = masks_to_outlines(masks)
    plt.imsave(saveas + ".png", outlines, cmap='gray')
    if not save_masks_only:
        imsave(saveas+'_id.png', masks)    
    print("masks saved to: "+saveas)


def mask2outline(mask_file):
    masks = imread(mask_file)
    outlines = masks_to_outlines(masks)
    skimage.io.imsave(os.path.splitext(mask_file)[0] + "_outline.png", img_as_ubyte(outlines))
    # imsave(os.path.splitext(mask_file)[0] + "_outline.png", outlines) # error
    # plt.imsave(os.path.splitext(mask_file)[0] + "_outline.png", outlines, cmap='gray') # saves as RGB file

# Coloring FP in mask map and FN in gt mask map
def color_fp_fn(mask_file, pred_file):
    mask = imread(mask_file)
    pred = imread(pred_file)
    mask_idx = np.setdiff1d(np.unique(mask), np.array([0])) # remove background 0
    pred_idx = np.setdiff1d(np.unique(pred), np.array([0])) # remove background 0
    
    iou = compute_iou(mask_true=mask, mask_pred=pred) # compute iou
    tp, fp, fn = tp_fp_fn(threshold=0.5, iou=iou, index=True)
    # tp_idx = mask_idx[tp]
    fp_idx = pred_idx[fp]
    fn_idx = mask_idx[fn]
    
    # plot fp with green in pred mask map
    total_idx = pred_idx
    pred_fp = pred.copy()
    for idx in total_idx:
        if(sum(idx == fp_idx) == 0):
            temp = np.where(pred_fp == idx)
            pred_fp[temp[0], temp[1]] = 0
    total_outlines = masks_to_outlines(pred)
    fp_outlines = masks_to_outlines(pred_fp)
    res = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    res[np.where(total_outlines)[0], np.where(total_outlines)[1], :] = 255
    res[np.where(fp_outlines)[0], np.where(fp_outlines)[1], 0] = 0
    res[np.where(fp_outlines)[0], np.where(fp_outlines)[1], 2] = 0
    plt.imsave(os.path.splitext(pred_file)[0] + "_outline_fp_green.png", res)
    
    # plot fn with green in gt mask map
    total_idx = mask_idx
    mask_fn = mask.copy()
    for idx in total_idx:
        if(sum(idx == fn_idx) == 0):
            temp = np.where(mask_fn == idx)
            mask_fn[temp[0], temp[1]] = 0
    total_outlines = masks_to_outlines(mask)
    fn_outlines = masks_to_outlines(mask_fn)
    res = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    res[np.where(total_outlines)[0], np.where(total_outlines)[1], :] = 255
    res[np.where(fn_outlines)[0],    np.where(fn_outlines)[1],    1] = 0
    res[np.where(fn_outlines)[0],    np.where(fn_outlines)[1],    2] = 0
    plt.imsave(os.path.splitext(pred_file)[0] + "_outline_fn_red.png", res)
    


def filter_by_intensity(image, masks, channels, min_avg_intensity=0, min_total_intensity=0):
    if(channels == [0,0]):
        im = image
    else:
        im = image[:,:,(channels[0]-1)]
                
    mask_indices = np.unique(masks)[1:]
    avg_intensities = ndimage.mean(im, labels=masks, index=mask_indices)
    total_intensities = ndimage.sum(im, labels=masks, index=mask_indices)
    
    indices_to_remove = mask_indices[np.where( (avg_intensities<min_avg_intensity) | (total_intensities<min_total_intensity) )[0]] # need to put () around boolean array before applying | 
    
    #for i in indices_to_remove: masks[np.where(masks==i)] = 0
    # this is more efficient
    masks[np.isin(masks, indices_to_remove)] = 0

    return masks
    

def GetCenterCoor(masks):
    # print(np.unique(masks))
    centers=ndimage.center_of_mass(masks, labels=masks, index=np.unique(masks)[1:]) # 1: to get rid of 0, which is background
    # centers=ndimage.center_of_mass(masks, labels=masks, index=list(range(1,np.max(masks)+1)))
    # alternative, slower
    # center_x=[]; center_y=[]
    # for i in range(1,ncell+1):
    #     mask_pixel = np.where(masks == i)
    #     center_y.append((np.max(mask_pixel[0]) + np.min(mask_pixel[0])) / 2)
    #     center_x.append((np.max(mask_pixel[1]) + np.min(mask_pixel[1])) / 2)
    
    return centers


def PlotMask_outline(mask, img, savefilename, color, fill=False):    
    if type(img)==str: img = imread(img)
    my_dpi = 96
    
    if fill:
        out_temp = (mask!=0)
        outX_temp, outY_temp = np.nonzero(out_temp)
    else:        
        out_temp = utils.masks_to_outlines(mask)
        outX_temp, outY_temp = np.nonzero(out_temp)
        
    if(img.ndim == 3):
        img[outX_temp, outY_temp] = np.array(color)
    elif(img.ndim == 2):
        zeros=np.zeros(img.shape, dtype='uint8')
        zeros[outX_temp, outY_temp] = 255
        img = np.stack([zeros, img, img], axis=-1)

    plt.figure(figsize=(mask.shape[1]/my_dpi, mask.shape[0]/my_dpi), dpi=my_dpi)
    plt.gca().set_axis_off()
    plt.imshow(img)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    # if(img.ndim == 2):
    #     plt.imsave(savefilename, imgout, cmap='gray')
    # if(img.ndim == 3):
    # imgout is always ndim=3 now
    plt.savefig(savefilename, bbox_inches = 'tight', pad_inches = 0)
    plt.close('all')


def PlotMask_center(mask, img, savefilename, color, add_text=False):
    if type(img)==str: img = imread(img)

    centers = GetCenterCoor(mask)
    y_coor, x_coor = zip(*centers); y_coor=list(y_coor); x_coor=list(x_coor)

    my_dpi = 96

    if(img.ndim == 3):
        imgout = img.copy()
    if(img.ndim == 2):
        zeros=np.zeros(img.shape, dtype='uint8')
        imgout = np.stack([zeros, img, img], axis=-1)

    plt.figure(figsize=(mask.shape[1]/my_dpi, mask.shape[0]/my_dpi), dpi=my_dpi)
    plt.gca().set_axis_off()
    plt.imshow(imgout)
    for i in range(len(centers)): 
    # for i in range(mask.max()): 
    # the followig won't work because max may be greater than the number of masks b/c some mask indices may be skipped
    # for i in range(len(np.unique(mask))-1): 
        if add_text:
            plt.text(x_coor[i], y_coor[i], str(i+1), dict(size=10, color='red', horizontalalignment='center', verticalalignment='center'))
        else:
            plt.plot(x_coor[i], y_coor[i], marker='o', color=color, ls='', markersize=4)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(savefilename, bbox_inches = 'tight', pad_inches = 0)
    plt.close('all')

    


def save_stuff(masks, imgfilename, channels, save_outlines_only=True, save_additional_images=False, save_mask_roi=False, img=None):
    if img is None: img = imread(imgfilename)
        
    filename = os.path.splitext(imgfilename)[0]
    
    outlines = utils.masks_to_outlines(masks)
    
    tmp=np.unique(masks, return_counts=True)
    sizes = tmp[1][1:]#.tolist()    # keep it as an array     
    mask_indices = tmp[0][1:]
    ncell=len(sizes)

    if(channels == [0,0]):
        im = img
    else:
        im = img[:,:,(channels[0]-1)]
    avg_intensities = ndimage.mean(im, labels=masks, index=mask_indices)
    total_intensities = ndimage.sum(im, labels=masks, index=mask_indices)

    centers=GetCenterCoor(masks)
    y_coor, x_coor = zip(*centers)
    # turn tuples into arrays to use as.type later
    y_coor=np.array(y_coor); x_coor=np.array(x_coor)

    # Save mask indices
    skimage.io.imsave(filename + "_masks_id.png", img_as_uint(masks), check_contrast=False)

    # Save mask outlines 
    if save_outlines_only:
        skimage.io.imsave(filename + "_masks.png", img_as_ubyte(outlines), check_contrast=False)
    else: 
        PlotMask_outline(mask=masks, img=img, savefilename=filename + "_masks.png", color=[255,0,0])        
    
    ## Save a csv file of mask info. One row per mask, columns include size, center_x, center_y
    mask_info = pd.DataFrame({
        "center_x": x_coor, 
        "center_y": y_coor,
        "size": sizes,
        "tfi": total_intensities,
        "mfi": avg_intensities
    })
    mask_info.index = [f"Cell_{i}" for i in range(1,ncell+1)]
    mask_info=mask_info.round().astype(int)
    mask_info.to_csv(filename + "_masks.csv", header=True, index=True, sep=',')
        
    # save _cp_outline to convert to roi by ImageJ
    if save_mask_roi:
        outlines_list = utils.outlines_list(masks)
        io.outlines_to_text(filename, outlines_list)
        
    if save_additional_images:     
        PlotMask_center(mask=masks, img=img, savefilename=filename + "_masks_point.png", color='r')
        PlotMask_center(mask=masks, img=img, savefilename=filename + "_masks_text.png",  color='r', add_text=True)
        skimage.io.imsave(filename + "_masks_fill.png", img_as_ubyte(masks!=0), check_contrast=False)
        # add image
        # PlotMask_outline(mask=masks, img=img, savefilename=filename + "_mask_fill.png", color=[255,255,255], fill=True)
