# Start from the rootproject/root-conda base image
FROM rootproject/root-conda:6.18.04

# Build the image as root user
USER root

# This sets the default working directory when a container is launched from the image
WORKDIR /home/docker

# Run as docker user by default when the container starts up
# USER docker

# Compile an executable named 'skim' from the skim.cxx source file
# RUN echo ">>> Compile skimming executable ..." &&  \
# COMPILER=$(root-config --cxx) &&  \
# FLAGS=$(root-config --cflags --libs) &&  \
# $COMPILER -g -std=c++11 -O3 -Wall -Wextra -Wpedantic -o skim skim.cxx $FLAGS
