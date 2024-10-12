import matplotlib.pyplot as plt
import os, cv2, warnings
import numpy as np
from scipy import ndimage
from tsp import imread
import skimage.io
from skimage import img_as_ubyte
from skimage.measure import regionprops


def dice(im1, im2):
    """
    Compute Dice coefficient between two binary masks.
    """
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

# def average_dice(gt_labels, pred_labels):
#     """
#     Compute average Dice coefficient per cell.
#     """
#     dice_scores = []
#     # Loop through unique labels (cells) in ground truth
#     for i in np.unique(gt_labels)[1:]:  # we ignore 0 as it's the background
#         gt_cell = gt_labels == i
#         overlap_with_pred = gt_cell * pred_labels  # element-wise multiplication
#         if sum(overlap_with_pred.flat):
#             pred_cell_label = np.bincount(overlap_with_pred.flat)[1:].argmax() + 1 # get the label of the largest overlapping predicted cell
#             pred_cell = pred_labels == pred_cell_label
#             dice_scores.append(dice(gt_cell, pred_cell))
#         else:
#             dice_scores.append(0)
    
#     return np.mean(dice_scores)

def average_dice(gt_labels, pred_labels):
    """
    Compute average Dice coefficient per cell.
    """

    dice_scores = []

    # Loop through regions in ground truth
    for region in regionprops(gt_labels):
        # Create a binary mask for the current region in ground truth
        gt_cell = gt_labels == region.label

        # Find the overlapping region in prediction
        pred_cell = pred_labels[region.coords[:, 0], region.coords[:, 1]]
        
        if sum(pred_cell):
            # Find the most common non-zero label in the overlapping region
            overlap_label = np.bincount(pred_cell[pred_cell > 0]).argmax()
            
            if overlap_label:  # ignore if there's no matching cell in prediction
                pred_cell_mask = pred_labels == overlap_label
                dice_scores.append(dice(gt_cell, pred_cell_mask))
            else:
                dice_scores.append(0)
                
        else:
            dice_scores.append(0)
                    
    return np.mean(dice_scores)


# IoU
def compute_iou(mask_true, mask_pred):
    '''
    Compute the IoU for ground-truth mask (mask_true) and predicted mask (mask_pred).
    '''
    true_objects = (np.unique(mask_true))
    pred_objects = (np.unique(mask_pred))
    
    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(mask_true, bins=np.append(true_objects, np.inf))[0]
    area_pred = np.histogram(mask_pred, bins=np.append(pred_objects, np.inf))[0]

    # Compute intersection between all objects
    # compute the 2D histogram of two data samples; it returns frequency in each bin
    # important to append n.inf otherwise the number of bins will be 1 less than the number of unique masks
    intersection = np.histogram2d(mask_true.flatten(), mask_pred.flatten(), bins=(np.append(true_objects, np.inf),np.append(pred_objects, np.inf)))[0] 
    
        
    area_true = np.expand_dims(area_true, -1) # makes true_objects * 1
    area_pred = np.expand_dims(area_pred, 0) # makes 1 * pred_objects
    
    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union
    return iou[1:, 1:] # exclude background; remove frequency for bin [0,1)


#    The following function is modified based on "_label_overlap()" and "_intersection_over_union" functions in cellpose github (https://github.com/MouseLand/cellpose/blob/main/cellpose/metrics.py).
#    For "intersection" below, the original functions seem not to deal with empty masks between background (value 0) and mask with maximum number (maximum value). It makes a difference between iou_map() and compute_iou() functions.
#    We modifed it so as to remove empty masks in the "intersection". After the modification, iou_map() and compute_iou() functions generates the same results.
'''
def iou_map(masks_ture, masks_pred):
    """IoU: Intersection over Union between true masks and predicted masks
       
    Inputs:
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels
    
    Outputs:
    iou: ND-array, float
        IoU map
    """
    x = masks_true.ravel() # flatten matrix to vector
    y = masks_pred.ravel() # flatten matrix to vector
    true_objects = masks_true.max()+1
    pred_objects = masks_pred.max()+1
    intersection = np.zeros((true_objects,pred_objects), dtype=np.uint)
    for i in range(len(x)):
        intersection[x[i], y[i]] += 1
    
    # modification #
    empty_mask_idx = []
    for i in range(intersection.shape[0]):
        if(sum(intersection[i,:]) == 0): empty_mask_idx.append(i)
    intersection = np.delete(intersection, empty_mask_idx, 0)
    
    n_pixels_true = np.sum(intersection, axis=1, keepdims=True)
    n_pixels_pred = np.sum(intersection, axis=0, keepdims=True)
    iou = intersection / (n_pixels_true + n_pixels_pred - intersection)
    iou[np.isnan(iou)] = 0.0
    return iou
'''


# TP, FP, FN
def tp_fp_fn(threshold, iou, index=False):
    '''
    Computes true positive (TP), false positive (FP), and false negative (FN) at a given threshold
    '''
    matches = iou >= threshold
    true_positives  = np.sum(matches, axis=1) >= 1 # predicted masks are matched to true masks
    false_positives = np.sum(matches, axis=0) == 0 # predicted masks are matched to false masks (number of predicted masks - TP)
    false_negatives = np.sum(matches, axis=1) == 0 # true masks are not matched to predicted masks (number of true masks - TP)
    if index:
        tp, fp, fn = (true_positives, false_positives, false_negatives)
    else:
        tp, fp, fn = (np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives))
    return tp, fp, fn

def tpfpfn(mask_true, mask_pred, threshold=0.5):
    iou = compute_iou(mask_true, mask_pred)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    return tp, fp, fn

# CSI
def csi(mask_true, mask_pred, threshold=0.5):
    '''
    Compute CSI (= TP/(TP+FP+FN)) at a given threshold
    '''
    iou = compute_iou(mask_true, mask_pred)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    csi = tp / (tp + fp + fn)
    return csi

# Precision
def precision(mask_true, mask_pred, threshold=0.5):
    '''
    Compute precision (= TP/(TP+FP) at a given threshold
    '''
    iou = compute_iou(mask_true, mask_pred)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    precision = tp / (tp + fp)
    return precision

# Recall
def recall(mask_true, mask_pred, threshold=0.5):
    '''
    Compute Recall (= TP/(TP+FN)) at a given threshold
    '''
    iou = compute_iou(mask_true, mask_pred)
    tp, fp, fn = tp_fp_fn(threshold, iou)
    recall = tp / (tp + fn)
    return recall

# Bias
def bias(mask_true, mask_pred):
    '''
    Compute Bias = (# of predicted masks / # of gt masks)-1
    '''
    gt_num = np.setdiff1d(np.unique(mask_true), np.array([0])) # remove background
    pred_num = np.setdiff1d(np.unique(mask_pred), np.array([0])) # remove background
    bias = (len(pred_num) / len(gt_num))-1
    return bias


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
    
    
    
color_dict = {
    "red":   np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue":  np.array([0, 0, 255]),
    # Add more color labels and RGB values as needed
}
# col is a list of keys from color_dict; if not supplied, white outlines will be saved
def mask2outline(mask_file, col=None):
    masks = imread(mask_file)
    if col is None:
        outlines = masks_to_outlines(masks)        
        with warnings.catch_warnings():
            # even when using img_as_ubyte, there may be low contrast warning if there are two few outlines, thus suppress warnings
            warnings.simplefilter("ignore", category=UserWarning)
            skimage.io.imsave(os.path.splitext(mask_file)[0] + "_outline.png", img_as_ubyte(outlines))
    else:
        colcode = [color_dict[i] for i in col]

        outlines = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)    
        slices = ndimage.find_objects(masks.astype(int))
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
                vr, vc = pvr + sr.start, pvc + sc.start 
                if i+1>len(colcode):
                    exit("run out of col code")
                else:
                    outlines[vr, vc, 0:3] = colcode[i]
        
        # need to use cvtColor b/c image is read with cellpose's imread, outputs image in RGB, but cv2.imwrite expects BGR
        cv2.imwrite(os.path.splitext(mask_file)[0] + "_outline.png", cv2.cvtColor(outlines, cv2.COLOR_BGR2RGB)) 


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
    

    
