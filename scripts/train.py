#!/usr/bin/env python

# import ROOT
import numpy as np
import keras
import tensorflow as tf
import uproot


filepath = "../atlas-open-data/mc_345323.VBFH125_WW2lep.exactly2lep.root"
treename = "mini"
nEvents = 100

f = uproot.open(filepath)
tree = f["mini"]

print("==>> Branch names:")
for i, n in enumerate(tree.allkeys()):
    print(n.decode('utf-8'))

branches = ["lep_pt", "jet_pt"]
# branches = [b.decode('utf-8') for b in tree.keys() if b.decode('utf-8') not in self.getBranchVetos()]

variables = tree.arrays(branches=branches, entrystop=nEvents, namedecode = "ascii")

# TODO: Think about variables in training!
# Construct variables! look at https://gitlab.cern.ch/kolehman/MVA-correlations/blob/master/training/Sample.py#L283


# Bring in nice numpy array format I guess?
    # def getInputForMVA(self):
    #     '''
    #     Returns a list of training variables. The last entry are the weights.
    #     '''
    #     MC = []
    #     for variable in self.getTrainingVariables():
    #         MC.append(self.__variables[variable])
    #     MC.append(self.__variables[self.getWeightName()])
    #     return MC, self.__nEvents

# shuffle
# np.random.shuffle(allMC)



print("End!")
