import os
import numpy as np
import cv2
import tifffile
from tqdm import tqdm


# copied from cellpose
# it use cv2.imread to read (for non-tiff files) and then converts the image from BGR to RGB 
# for png files, it is closest to skimage.io.imread
def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]['shape']
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                #io_logger.info(f'reading tiff with {ltif} planes')
                img = np.zeros((ltif, *shape), dtype=dtype)
                for i,page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)            
        return img
    
    elif ext != '.npy':
        # png, jpg, etc
        try:
            # -1 means as is and keeps alpha channel for png
            img = cv2.imread(filename, -1)
            # this is needed because cv2 uses BGR instead of RGB
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            print('ERROR: could not read file, %s'%e)
            return None
    
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat['masks']
            return masks
        except Exception as e:
            print('ERROR: could not read masks from file, %s'%e)
            return None

def imsave(filename, arr):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        tifffile.imsave(filename, arr)
    else:
        # assume arr is RGB
        cv2.imwrite(filename, arr)

# an alias for imsave
def imwrite(filename, arr):
    imsave(filename, arr)


def normalize99(Y, lower=1,upper=99):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X

def normalize100(Y, lower=0,upper=100):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X

def image_to_rgb(img0, channels=[0,0]):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = np.clip(normalize100(img[:,:,i]), 0, 1) # replacing normalize99 with normalize100 makes a big difference for some images
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB

