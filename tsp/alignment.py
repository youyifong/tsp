# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:49:24 2023

@author: Youyi
"""
import os, cv2, sys
import numpy as np
from tsp import imsave, imread

def doalign (ref_image, image2, channels, alignmentmode):
    filename, file_extension = os.path.splitext(image2)

    image1=imread(ref_image)
    image2=imread(image2)

    if (image1.ndim==3 or image2.ndim==3) and channels is None:
        sys.exit("When at least one of the two images is 3-dimensional, --l needs to be set")

    # Convert images to grayscale for computing the rotation via ECC method
    if len(image1.shape)==3: image1 = image1[:,:,channels[0]]
    im1_gray = image1    
    if len(image2.shape)==3: image2 = image2[:,:,channels[1]]
    im2_gray = image2

    # motion models: MOTION_HOMOGRAPHY, MOTION_AFFINE, MOTION_EUCLIDEAN, MOTION_TRANSLATION
    if alignmentmode == "MOTION_HOMOGRAPHY":
        warp_mode = cv2.MOTION_HOMOGRAPHY
    elif alignmentmode == "MOTION_AFFINE":
        warp_mode = cv2.MOTION_AFFINE
    elif alignmentmode == "MOTION_EUCLIDEAN":
        warp_mode = cv2.MOTION_EUCLIDEAN
    elif alignmentmode == "MOTION_TRANSLATION":
        warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(3 if warp_mode == cv2.MOTION_HOMOGRAPHY else 2, 3, dtype=np.float32)
    
    number_of_iterations = 15000;         
    termination_eps = 1e-2; # Specify the threshold of the increment in the correlation coefficient between two iterations
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 1)
             
    sz = image1.shape    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        image2_warp = cv2.warpPerspective (image2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP); # INTER_LINEAR or INTER_NEAREST
    else :        
        image2_warp = cv2.warpAffine      (image2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP);
    
    imsave(filename+"_aligned"+file_extension,  image2_warp)

     
    ## elastix outcomes are blurred. In histograms, the background value 0 is not a peak
    # # Convert the images to gray level: color is not supported. This step can be changed to average across channels
    # image1 = rgb2gray(image1)
    # image2 = rgb2gray(image2)
    # # image1 = image1[:,:,0]
    # # image2 = image2[:,:,0]
    
    # # Get params and change a few values
    # params = pyelastix.get_default_params(type="RIGID")
    # params.Transform="SimilarityTransform"
    # params.Interpolator = "NearestNeighborInterpolator"   
    # params.NumberOfResolutions = 1
    # # params.FixedInternalImagePixelType = "int"
    # # params.MovingInternalImagePixelType = "int"
    
    # # params.ResampleInterpolator = "NearestNeighborResampleInterpolator" # component not installed
    # # params.MaximumNumberOfIterations = 200
    # # params.FinalGridSpacingInVoxels = 10
    
    # # Apply the registration (im1 and im2 can be 2D or 3D)
    # image2_warp, field = pyelastix.register(image2, image1, params)

    ## check the frequency of 0 in intensity distribution
    # print(np.histogram(image2_warp, bins=np.append(np.unique(image2_warp), np.inf)))
            
    # save to file
    # print(image2_warp.shape) %3D image after cv2
    # image2_warp=image_to_rgb(image2_warp) # now 3D
    # print(np.histogram(image2_warp, bins=np.append(np.unique(image2_warp), np.inf)))

        