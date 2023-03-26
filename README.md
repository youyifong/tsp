# TSP - The Seattle Pipeline for Deep Learning Methods for Cell and Tissue Imaging Analysis

## Pipeline steps

Pre-processing

- background reduction and optimizing brightness using the ZKW DataWizard application
- eliminating/blacking out false or artificial fluorescence signals generated by debris introduced into the chip chamber during the multiple staining cycles using Fiji (ImageJ)
- alignment using tsp
- cropping


Analysis

- cell segmentation
- compute distance to boundary
- compute region membership
- statistical analysis

## Image alignment

> python -m tsp alignimages --ref_image xx  --image2 xx --l [2,2]

The --l argument is used to extract the channels of interest from the two images. 
In the example above, the second channel of image2 will be aligned against the second channel of ref_image.

A new image file named _aligned.png will be saved.

 
## Working with masks

To convert roi files into mask png files
> python -m tsp roifiles2mask --roifolder eg1 --height 1040 --width 1392 
 
where roifolder is the path to the folder containing the unzipped roi files, height and width are the dimension of the image. The program creates two png files, one mask file and one mask outline file. 

To unzip, e.g.

> unzip CF_Les_Pos7_CD3+CD8+_RoiSet_865.zip -d CF_Les_Pos7_CD3+CD8+_RoiSet_865 

To compare two mask files to get AP
> python -m tsp AP --mask1 testmasks/M872956_JML_Position10_CD3_test_masks.png --mask2  testmasks/M872956_JML_Position10_CD3_test_masks.png 
> 
> python -m tsp AP --mask1 M926910_Position1_CD45+CD56+_seg.npz --mask2 M926910_Position1_CD45+CD56+CD3-CD271-_seg.npz 

To compare two folders of masks
> python -m tsp checkprediction --metric   --predfolder   --gtfolder   --min_size

To add mask1 in red, mask2 in green (optional), and overlap in yellow, all on top of images
> python -m tsp overlaymasks

To add mask2 in green and highlight tp (based on comparing with mask1) in yellow, on top of images
> python -m tsp colortp

To make outlines
> python -m tsp maskfile2outline --maskfile 


## Cell segmentation with cellpose 
> python -m tsp runcellpose --f '*.png' 

Required:
- --f is required and tells the program which image files to segment. The quotes around file name pattern are required, because otherwise it will be expanded by shell

- --l Signal channels. The channels have the format as [cytoplasm,nucleus], and each value can be 0 (grayscale), 1 (red), 2 (green), and 3 (blue). Default channels are [3,0] that means blue cytoplasm and no nuclei. E.g., -l=[0,0] if image is grayscale. 

- --model Cellpose model to use, including cyto and cytotrain7. cyto is the Cellpose default, cytotrain7 is the model from Sunwoo et al.

Optional:
- --s If present, some additional image files will be saved.

- --saveflow If present, .npy file containing flow, diam etc will be saved.

- --saveroi If present, mask outline roi files will be saved.

- --d Cellpose tuning parameter, cell diameter. Default 0. 

- --o Cellpose tuning parameter, flow threshold. Default 0.4.  

- --cellprob Cellpose tuning parameter, cellprob threshold. Default 0. 

- --min_size Post-processing parameter, min_size, is changed from 15 (default) to the specified value. If a cell consists of the number of pixels less than min_size, the cell is removed. 

- --min_average_intensity Post-processing parameter, min_average_intensity, is changed from 0 (default) to the specified value. If an average intensity of pixels in a cell is less than min_average_intensity, the cell is removed. 

- --min_average_intensity Post-processing parameter, minimum total intensity, is changed from 0 (default) to the specified value. If the total intensity of pixels in a cell is less than min_total_intensity, the cell is removed. 


#### Output 
- cellpose_counts_timestr.txt: number of predicted masks for each image 

- _masks.csv: a text file containing info about the size of each predicted mask and x-y coordinate of center pixel of each predicted mask  

- _mask_fill.png (with -s): an image file containing the solid fill of the predicted cell masks 

- _mask_outline.png (with -s): an image file containing the predicted cell masks 

- _mask_point.png (with -s):  an image file containing a point indicating the center of the predicted cell masks 

- _mask_text.png (with -s): an image file containing the identified numbers of the predicted cell masks 

- _seg.npy (with -saveflow): cellpose output file containing overall info about predicted masks and parameters used in prediction. This file can be huge.

- _cp_outlines.txt (with -saveroi): a text file that can be converted to roi by ImageJ if imagej_roi_converter.py from Cellpose is run as a macro after opening image file


## Cell phenotyping 
> python -m tsp cellphenotyping --f [file1.png,file2.png,file3.png] --m [Mask,Mask] --c [0.5,0.5] --p [True,False] --n [marker2,marker3]

The command above looks for marker1+marker2+marker3- cells.  Let K be the number of markers. 

- --f Required. List of K file names. In this example, for the first file, the program expects to find both file1.png and file1_seg.npy. For the following files, e.g. file2.png, the program expects to find the mask file (_seg.npy) if the method is Mask and the image file otherwise. 

- --m Required. List of K-1 values from Mask, Intensity_avg_pos, Intensity_avg_all, or Intensity_total. Methods for finding multistained cells. Under mask, overlap between A and B is computed for individual B cells, not all B cells.

- --c Required. List of K-1 cutoff values for deciding if markers are present.

- --p Required. List of K-1 values from True or False. Marker is required to be present if True and abscent if False.

- --n Required. List of K-1 names for markers (first marker excluded). 

- --l Signal channels. The channels have the format as [cytoplasm,nucleus], and each value can be 0 (grayscale), 1 (red), 2 (green), and 3 (blue). Default channels are [3,0] that means blue cytoplasm and no nuclei. E.g., -l=[0,0] if image is grayscale. 

- --r If present, two additional result files will be saved for the last stage: 1) a cellpose output file named _seg.npz that contains information of masks, outlines, flows, and a cell diameter, 2) a simple text file named _masks.csv that contains the sizes and the x and y coordinates for each mask. 

#### Output 
- _counts_multistain.txt: cell counts, one row for each additional marker

- _counts_lastcutoff.txt: cell counts for a series of cutoffs for the last marker

- _masks.csv: a text file containing info about the size of each predicted mask and x-y coordinate of center pixel of each predicted mask, last marker only

- _fill.png (with -s): K files with each mask drawn as a filled shape. The underlying image is the kth marker image despite the file name.

- _outline.png (with -s): K files with each mask drawn as an outline. The underlying image is the kth marker image despite the file name.

- _point.png (with -s): K files with each mask drawn as a point. The underlying image is the kth marker image despite the file name.

- _seg.npz (with -r): cellpose output file containing overall info about image and masks, last marker only




## Intensity statistics 

> python -m tsp intensityanalysis --f [file1.png,file2.png,file3.png] 

Measures the intensities of all markers in each marker 1 mask.

- --f Required. List of K file names. In this example, for the first file, the program expects to find both file1.png and file1_seg.npy. For the following files, e.g. file2.png, the program expects to find the image file. 

- --l Signal channels. The channels have the format as [cytoplasm,nucleus], and each value can be 0 (grayscale), 1 (red), 2 (green), and 3 (blue). Default channels are [3,0] that means blue cytoplasm and no nuclei. E.g., -l=[0,0] if image is grayscale. 

#### Output 
- _intensity.txt: contains three intensity (total intensity, average normalized intensity, and total normalized intensity) and the x-y coordinates for each mask of marker 1



## Compute distance from cell center to a boundary polyline

Appends a dist2boundary column of the shortest distance from cell centers to the boundary to the input csv file and save as a new _d2b.csv file.

> python -m tsp adddist2line --cells masks.csv  --boundaryroi boundary.roi

- --cells: a csv file containing the cell center coordinates

- --boundaryroi: an roi file defining the boundary



## Compute region membership


> python -m tsp regionmembership --cells masks.csv  --regionroi [region1.roi,region2.roi]

- --cells: a csv file containing the cell center coordinates

- --boundaryroi: an roi file defining the boundary

 

