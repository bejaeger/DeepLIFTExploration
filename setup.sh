#!/bin/bash

echo "Calling setup.sh script"

# setup script for a few shortcuts
alias pullimg='docker pull gitlab-registry.cern.ch/bejaeger/deepliftforhep'
alias runcont='docker run --rm -it -v $PWD:/home/docker/DeepLIFTforHEP gitlab-registry.cern.ch/bejaeger/deepliftforhep:latest /bin/bash'
alias buildimglocal='docker build -f Dockerfile -t deepliftforhep:latest --rm=true --force-rm=true .'
alias runcontlocal='docker run --rm -it -v $PWD:/home/docker/DeepLIFTforHEP deepliftforhep:latest /bin/bash'

export INPUTDIR=$PWD/../atlas-open-data/
export OUTPUTDIR=$PWD/output/
