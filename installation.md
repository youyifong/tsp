# Setup Instructions for Running The Seattle Pipeline  

 
## Initially

Load modules  

These instructions are specific to the Hutch cluster users. If these two lines are added to .profile (assuming bash shell is used), there will be no need to run them manually. 

>ml Python/3.9.6-GCCcore-11.2.0 

>ml cuDNN/8.2.2.26-CUDA-11.4.1 

  

Create a virtual env on Linux 

optionally, -p /app/software/Python/3.9.6-GCCcore-11.2.0/bin/python can be added to the virtualenv command 

>mkdir tspenv 

>virtualenv tspenv    

>source tspenv/bin/activate 

 
Clone our tsp pipeline package and install with the editable –e option 

>git clone https://github.com/youyifong/tsp 

>pip install -e tsp 
 
 
Install our fork of Cellpose (could take some time)

>pip install git+https://github.com/youyifong/cellpose.git#egg=cellpose 


Install additional python packages 

>pip install matplotlib read_roi pytz 
 

## Refresh

 
To refresh tsp, simply enter the tsp directory and that will update the installed module

>git pull 

To refresh Cellpose, 

>pip install git+https://github.com/youyifong/cellpose.git#egg=cellpose  

 
## Some tips 

It is good to work within python virtual env. To activate python virtual environment, source tspenv/bin/activate. Once activated, we will see the virtual env name appearing before the prompt. 

To prevent loss of work when the ssh connection is interrupted, run nohup python ... &, which will continue running even in the case of a network interruption. 

 

 
## Hardware environment at the Hutch 

The Hutch Scientific Computing (Scicomp) has nodes with GPUs in the gizmo cluster. They are shared across the center, so the ability to get one of these servers depends on the length of the queue. Scicomp also helps maintain private servers and we own one of those, named volta. Volta has a different software stack from the gizmo cluster, so our ability to continue using volta depends on Scicomp’s continued support of the volta machine. 

To access volta, open a terminal and run ssh username@volta. For example, if your HutchNet ID is abc, you can run ssh abc@volta. For details, you can refer to https://sciwiki.fredhutch.org/scicomputing/access_credentials 

Given the size of the images we can use the Fast storage under /fh/fast/fong_y (https://sciwiki.fredhutch.org/scicomputing/store_posix/). 

 