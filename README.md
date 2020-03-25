# DeepLIFTforHEP

Code repository for the CMPT726 project in Spring 2020

# Run it

```bash
# clone gitlab repository 
git clone https://gitlab.cern.ch/bejaeger/DeepLIFTforHEP.git
cd DeepLIFTforHEP
source setup.sh # defines a few aliases and env variables

# pull docker image uploaded docker image
pullimg # pull latest container from gitlab
runcont # runs the container

# inside container, run visualize script
source setup.sh # manually sourcing is currently still needed
cd scripts
./visualize.py # this should produce a test png file in a new directory called 'output/'.
```
