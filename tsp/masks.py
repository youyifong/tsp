import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_roi import read_roi_file, read_roi_zip
from PIL import Image, ImageDraw
from scipy import ndimage
from cellpose import utils, io
from tsp import imread, imsave
import skimage.io
from skimage import img_as_ubyte, img_as_uint
from tsp.AP import masks_to_outlines
import sys


# From .roi files to masks file
def roifiles2mask(files, width, height, saveas):
    masks = Image.new('I', (width, height), 0)
    
    _, extension = os.path.splitext(files[0])

    if extension == ".zip":
        if len(files)>1:
            sys.exit("one zip file at a time")            
        rois = read_roi_zip(files[0])
    else:
        rois = []        
        for idx in range(len(files)):
            roi_file = files[idx]
            rois.append(read_roi_file(roi_file))
        
    print("number of roi files: "+str(len(rois)))
    
    if extension == ".zip":
        keys = list(rois.keys())
    
    
    for idx in range(len(rois)):
        
        if extension == ".zip":
            mask_temp = rois[keys[idx]]                
        else:
            roi_file = files[idx]
            filename = roi_file.split(os.sep)[-1][:-4]
            mask_temp = rois[idx][filename]
                
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

    filename=os.path.splitext(saveas)[0]
    fileext=os.path.splitext(saveas)[1]
    imsave(filename+"_m"+fileext, masks)    
    
    # save an outline file
    outlines = masks_to_outlines(masks)
    plt.imsave(filename+"_o"+fileext, outlines, cmap='gray')
    
    print("masks saved to: "+saveas)



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

    


def save_stuff(masks, imgfilename, model, channels, save_outlines_only=True, save_additional_images=False, save_mask_roi=False, img=None):
    if img is None: img = imread(imgfilename)
        
    filename = os.path.splitext(os.path.basename(imgfilename))[0]
    
    if len(model)>5:
        # model is a long file name
        model='new'
    
    outlines = utils.masks_to_outlines(masks)
    
    tmp=np.unique(masks, return_counts=True)
    sizes = tmp[1][1:]#.tolist()    # keep it as an array     
    mask_indices = tmp[0][1:]
    ncell=len(sizes)
    print(f"number of cells: {ncell}")

    if(channels == [0,0]):
        im = img
    else:
        im = img[:,:,(channels[0]-1)]
    
    print(im.shape)
    print(masks.shape)
    print(mask_indices)
    
    avg_intensities = ndimage.mean(im, labels=masks, index=mask_indices)
    median_intensities = ndimage.median(im, labels=masks, index=mask_indices)
    total_intensities = ndimage.sum(im, labels=masks, index=mask_indices)

    centers=GetCenterCoor(masks)
    y_coor, x_coor = zip(*centers)
    # turn tuples into arrays to use as.type later
    y_coor=np.array(y_coor); x_coor=np.array(x_coor)

    # Save mask indices
    skimage.io.imsave(filename + "_m_"+model+".png", img_as_uint(masks), check_contrast=False)

    # Save mask outlines 
    if save_outlines_only:
        skimage.io.imsave(filename + "_o_"+model+".png", img_as_ubyte(outlines), check_contrast=False)
    else: 
        PlotMask_outline(mask=masks, img=img, savefilename=filename + "_o_"+model+".png", color=[255,0,0])        
    
    ## Save a csv file of mask info. One row per mask, columns include size, center_x, center_y
    mask_info = pd.DataFrame({
        "center_x": x_coor, 
        "center_y": y_coor,
        "size": sizes,
        "tfi": total_intensities,
        "medfi": median_intensities,
        "mfi": avg_intensities
    })
    mask_info.index = [f"Cell_{i}" for i in range(1,ncell+1)]
    mask_info=mask_info.round().astype(int)
    mask_info.to_csv(filename + "_m_"+model+".csv", header=True, index=True, sep=',')
        
    # save _cp_outline to convert to roi by ImageJ
    if save_mask_roi:
        outlines_list = utils.outlines_list(masks)
        io.outlines_to_text(filename, outlines_list)
        
    if save_additional_images:     
        PlotMask_center(mask=masks, img=img, savefilename=filename + "_point_"+model+".png", color='r')
        PlotMask_center(mask=masks, img=img, savefilename=filename + "_text_"+model+".png",  color='r', add_text=True)
        skimage.io.imsave(filename + "_fill_"+model+".png", img_as_ubyte(masks!=0), check_contrast=False)
        # add image
        # PlotMask_outline(mask=masks, img=img, savefilename=filename + "_fill_"+model+".png", color=[255,255,255], fill=True)
