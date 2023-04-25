# Deep learning based denoising process

# Reference  
Yoon, Taekeun, et al. "Deep learning-based denoising for fast time-resolved flame emission spectroscopy in high-pressure combustion environment." Combustion and Flame 248 (2023): 112583.
https://doi.org/10.1016/j.combustflame.2022.112583

# Including
models	 	: model strucutre python code
utils 		: required functions 
enivornment.yaml 	: conda environment
DU_CNN.py	: main code
data		: raw dataset (download following url https://drive.google.com/file/d/1yOuxJmI4tKYI3tJEJIWKf52T4SjAfaSB/view?usp=share_link
rawdataM.mat' and locate in data folder)

# Requirement
Visual studio code
python environment (anaconda)
python version 3.10.4
$ : command 

# Procedure
1. Select working directory
2. $ conda create -n n2s
3. Change prefix in enviornment.yami (bottom line)
4. $ conda activate n2s
5. $ conda env create --file environment.yaml
6. $ python DU_CNN.py


# tip: Install the required libraries if you get the error
 
# Note1
If the GPU memory is not enough,
Reduce batch_size

# Note2
Change the model structure using following parameter
    layerss=[5,11]
    kernelss=[7,15,25]
    channelss=[32]
    downsampless=[1,16]  
