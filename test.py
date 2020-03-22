#!/usr/bin/env python
import ROOT
import os

import sys
print(sys.version_info)

f = ROOT.TFile.Open("../atlas-open-data/mc_345323.VBFH125_WW2lep.exactly2lep.root")
t = f.Get("mini")

print("First 10 branch names:")
for i, n in enumerate(t.GetListOfBranches()):
    print(n)
    if i >= 10: break

for i, e in enumerate(t):
    print("Event = {}: nJets = {}".format(i, e.jet_n))
    if i >= 10: break
    
print("End!")
