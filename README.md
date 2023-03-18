# TSP - The Seattle Pipeline for Deep Learning Methods for Cell and Tissue Imaging Analysis


## To convert roi files into mask png files: 

> python -m tsp roifiles2mask --roifolder eg1 --height 1040 --width 1392 

Where roifolder is the path to the folder containing the unzipped roi files, height and width are the dimension of the image. The program creates two png files, one mask file and one mask outline file. 

To unzip, e.g.: 

unzip CF_Les_Pos7_CD3+CD8+_RoiSet_865.zip -d CF_Les_Pos7_CD3+CD8+_RoiSet_865 


## To align images: 

> python -m tsp alignimages --ref_image xx  --image2 xx 

 
