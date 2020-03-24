# DeepLIFTforHEP

Code repository for the CMPT726 project in Spring 2020

# Run it

```bash
# clone gitlab repository 
git clone https://gitlab.cern.ch/bejaeger/DeepLIFTforHEP.git
cd DeepLIFTforHEP
source setup.sh # defines a few aliases and env variables

# run uploaded docker image
runimg # runs the container
source setup.sh # manually sourcing currently needed

# run visualize test script (inside container)
cd scripts
./visualize.py # this should produce a test png file in a new directory called 'output/'.
```
