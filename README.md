# DeepLIFTforHEP

Code repository for the CMPT726 project in Spring 2020

# Run it

```
# clone and change dir
git clone https://gitlab.cern.ch/bejaeger/DeepLIFTforHEP.git
cd DeepLIFTforHEP

# run uploaded docker image 
docker run --rm -it -v $PWD:/home/docker/DeepLIFTforHEP gitlab-registry.cern.ch/bejaeger/deepliftforhep:latest /bin/bash

# run visualize test script (inside container)
cd scripts
./visualize.py # this should produce a test png file in the current directory 
```
