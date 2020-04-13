# DeepLIFTforHEP

Code repository for the CMPT726 project in Spring 2020

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
