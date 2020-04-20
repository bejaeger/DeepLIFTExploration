# DeepLIFTforHEP

Code repository for the CMPT726 project in Spring 2020.
Docker needs to be installed to be able to run the analysis of the ATLAS Open Data. 

# Quick Start

```bash
# clone gitlab repository 
git clone https://gitlab.cern.ch/bejaeger/DeepLIFTforHEP.git
cd DeepLIFTforHEP
source setup.sh # defines a few aliases and env variables

# pull docker image uploaded docker image
pullimg # pull latest container from gitlab
# this runs the alias: 'docker pull gitlab-registry.cern.ch/bejaeger/deepliftforhep'
# you might need to execute docker as superuser with sudo: 'sudo docker ...'
runcont # runs the container
# this runs the alias: 'docker run --rm -it -v $PWD:/home/DeepLIFTforHEP gitlab-registry.cern.ch/bejaeger/deepliftforhep:latest /bin/bash'

# inside container, run visualize script
source setup.sh # manually sourcing is currently still needed
cd scripts
./visualize.py
# this produces distributions of the input variables used in the DNN training in a new directory called 'output/'.
```



1) MNIST subfolder

mnist_original_cnn.py - trains a CNN with parameters identical to original DeepLIFT paper and saves model into mnist_my_cnn_model.h5
mist_modified_cnn.py - trains a CNN with modified architecture (see report) and saves model into mnist_my_cnn_model.h5 (included)
mnist_test.py - runs DeepLIFT on the model in mnist_my_cnn_model.h5 and produces plots as in original DeepLIFT paper

2) Two Moons subfolder

TwoMoonsModel.h5 - trained model for Two Moons classfication (see report)

two_moons.py - traines a simple NN for Two Moons classification and runs DeepLIFT. Produces plots in report.
TwoMoonsTest.ipynb - notebook version