# DeepLIFTforHEP

Code repository for the SFU CMPT726 project in Spring 2020.
An exploration of the [DeepLIFT](https://github.com/kundajelab/deeplift) (Deep Learning Important FeaTures) algorithm in three steps.

## MNIST subfolder

`mnist_original_cnn.py`: trains a CNN with parameters identical to original DeepLIFT paper and saves model into mnist_my_cnn_model.h5     
`mist_modified_cnn.py:` trains a CNN with modified architecture (see report) and saves model into mnist_my_cnn_model.h5 (included)     
`mnist_test.py`: runs DeepLIFT on the model in mnist_my_cnn_model.h5 and produces plots as in original DeepLIFT paper     

## Two Moons subfolder

`TwoMoonsModel.h5`: trained model for Two Moons classfication (see report)

`two_moons.py`: traines a simple NN for Two Moons classification and runs DeepLIFT. Produces plots in report.
`TwoMoonsTest.ipynb`: notebook version

## ATLAS subfolder
For running the analysis on the ATLAS Open Data a docker image needs to be downloaded

### Download and run the docker image
```bash
# In the main DeepLIFTforHEP directory:
source setup.sh # defines a few aliases and env variables
# pull docker image uploaded docker image
pullimg # pull latest container from gitlab with ATLAS Open Data included (takes a while)
# this runs the alias: 'docker pull gitlab-registry.cern.ch/bejaeger/deepliftforhep'
# you might need to execute docker as superuser with sudo: 'sudo docker ...'
runcont # runs the container
# this runs the alias: 'docker run --rm -it -v $PWD:/home/DeepLIFTforHEP gitlab-registry.cern.ch/bejaeger/deepliftforhep:latest /bin/bash'

# inside container you have to manually source the setup.sh again
source setup.sh # manually sourcing is currently still needed
cd ATLAS #
./visualize.py # run script to produce plots as an example
```

#### Main scripts
`visualize.py`: produces plots of the input variable distributions in a new directory called 'output/'      
`train.py`: trains an NN to classify signal and background events          
`analyze.py`: applies DeepLIFT to the trained NN and saves plots of importances scores         

#### Helper files/modules
`config.cfg`: settings for the main scripts         
`plot.py`: plot helpers         
`helper.py`: miscellaneous helper functions           
`loaddata.py`: functions for I/O        
`variables.py`: class and functions to modify/constructinput variables        

