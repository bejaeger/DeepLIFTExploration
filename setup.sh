#!/bin/bash

echo "Calling setup.sh script"

# setup script for a few shortcuts
alias runimg='docker run --rm -it -v $PWD:/home/docker/DeepLIFTforHEP gitlab-registry.cern.ch/bejaeger/deepliftforhep:latest /bin/bash'
alias buildimglocal='docker build -f Dockerfile -t deepliftforhep:latest .'
alias runimglocal='docker run --rm -it -v $PWD:/home/docker/DeepLIFTforHEP deepliftforhep:latest /bin/bash'


