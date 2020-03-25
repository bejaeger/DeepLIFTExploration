# Start from the rootproject/root-conda base image with ROOT and python3.7
FROM rootproject/root-conda:6.18.04

# Build the image as root user
USER root

# This sets the default working directory when a container is launched from the image
WORKDIR /home/

# get samples in sample directory
RUN wget https://cernbox.cern.ch/index.php/s/e4A8pcCmzBei3Pc/download --output-document=mc_345323.VBFH125_WW2lep.exactly2lep.root
# RUN wget https://cernbox.cern.ch/index.php/s/UAf9leEd6nf6t1N/download --output-document=mc_363492.llvv.exactly2lep.root
RUN wget https://cernbox.cern.ch/index.php/s/C0gb451mNtemENl/download --output-document=mc_410000.ttbar_lep.exactly2lep.root
RUN mkdir atlas-open-data
RUN mv mc_* atlas-open-data

# Install some necessary software
RUN pip install --upgrade pip setuptools && \
     pip install tensorflow && \
    pip install keras && \
    pip install uproot && \
    pip install matplotlib && \
    pip install pandas && \
    pip install atlasify && \ 
    pip install sklearn

# Run as docker user by default when the container starts up
# USER docker

RUN echo 'export PS1="\[\033[01;31m\][Container] \[\033[00m\]${debian_chroot:+($debian_chroot)}\[\033[0;31m\]\u@\h\[\033[37m\]:\[\033[01;37m\]\w\[\033[00m\] "' >> /root/.bashrc

# change workdir to DeepLIFTforHEP
WORKDIR /home/DeepLIFTforHEP

# Compile an executable named 'skim' from the skim.cxx source file
# RUN echo ">>> Compile skimming executable ..." &&  \
# COMPILER=$(root-config --cxx) &&  \
# FLAGS=$(root-config --cflags --libs) &&  \
# $COMPILER -g -std=c++11 -O3 -Wall -Wextra -Wpedantic -o skim skim.cxx $FLAGS

