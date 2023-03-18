# TSP - The Seattle Pipeline for Deep Learning Methods for Cell and Tissue Imaging Analysis

## To align images: 

> python -m tsp alignimages --ref_image xx  --image2 xx 

 
## Working with masks

### To convert roi files into mask png files: 
> python -m tsp roifiles2mask --roifolder eg1 --height 1040 --width 1392 
> 
Where roifolder is the path to the folder containing the unzipped roi files, height and width are the dimension of the image. The program creates two png files, one mask file and one mask outline file. 

To unzip, e.g.

> unzip CF_Les_Pos7_CD3+CD8+_RoiSet_865.zip -d CF_Les_Pos7_CD3+CD8+_RoiSet_865 

### To compare two mask files to get average precision (AP)
> python -m tsp AP --mask1 testmasks/M872956_JML_Position10_CD3_test_masks.png --mask2  testmasks/M872956_JML_Position10_CD3_test_masks.png 
> 
> python -m tsp AP --mask1 M926910_Position1_CD45+CD56+_seg.npz --mask2 M926910_Position1_CD45+CD56+CD3-CD271-_seg.npz 

### To compare two folders of masks
> python -m tsp checkprediction --metric   --predfolder   --gtfolder   --min_size

### To add mask1 in red, mask2 in green (optional), and overlap in yellow, all on top of images
> python -m tsp overlaymasks

### To add mask2 in green and highlight tp (based on comparing with mask1) in yellow, on top of images
> python -m tsp colortp

### To makes masks png file
> python -m tsp roifiles2mask --roifolder   --width   --height  

### To make outlines
> python -m tsp maskfile2outline --maskfile 


## To run cellpose to do cell segmentation: 
> python -m tsp runcellpose --f '*.png' 

Output 
- cellpose_counts_timestr.txt: number of predicted masks for each image 

- _mask_fill.png (with -s): an image file containing the solid fill of the predicted cell masks 

- _mask_outline.png (with -s): an image file containing the predicted cell masks 

- _mask_point.png (with -s):  an image file containing a point indicating the center of the predicted cell masks 

- _mask_text.png (with -s): an image file containing the identified numbers of the predicted cell masks 

- _seg.npy (with -r): cellpose output file containing overall info about predicted masks and parameters used in prediction 

- _sizes_coordinates.txt (with -r): a text file containing info about the size of each predicted mask and x-y coordinate of center pixel of each predicted mask  

Option: 
- --f is required and tells the program which image files to segment. The quotes around file name pattern are required.  

- --p Cellpose models. The default is 'cytotrain7' to use the trained model on seven training images from K. This can be changed to 'tissuenet' to use the trained model on tissuenet images and to 'cyto' to use the pre-trained cellpose model on cytoplasm cellpose images. The trained models are saved under /fh/fast/fong_y/ cellpose_trained_models/  

- --d Cellpose tuning parameter, cell diameter. Default is good. 

- --o Cellpose tuning parameter, flow threshold. Default is good.  

- --c Cellpose tuning parameter, cellprob threshold. Default is good. 

- --l Signal channels. The channels have the format as [cytoplasm,nucleus], and each value can be 0 (grayscale), 1 (red), 2 (green), and 3 (blue). Default channels are [3,0] that means blue cytoplasm and no nuclei. E.g., -l=[0,0] if image is grayscale. 

- --min_size Post-processing parameter, min_size, is changed from 15 (default) to the specified value. If a cell consists of the number of pixels less than min_size, the cell is removed. 

- --min_average_intensity Post-processing parameter, min_average_intensity, is changed from 0 (default) to the specified value. If an average intensity of pixels in a cell is less than min_average_intensity, the cell is removed. 

- --min_average_intensity Post-processing parameter, minimum total intensity, is changed from 0 (default) to the specified value. If the total intensity of pixels in a cell is less than min_total_intensity, the cell is removed. 

- --s Four image files will be saved. 1) _mask_outline.png that contains cell masks, 2) _mask_text.png that contains the identified numbers of cell masks, 3) _mask_point.png that contains the center point of cell masks, 4) _mask_fill.png that contains the solid fill of cell masks 

- --r Two resulting files will be saved. 1) a cellpose output file named _seg.npy that contains information of masks, outlines, flows, and a cell diameter, 2) a simple text file named _sizes_coordinates.txt that contains the sizes and the x and y coordinates for each mask. 


