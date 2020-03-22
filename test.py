import ROOT
import os



f = ROOT.TFile.Open("atlas-open-data/mc_345323.VBFH125_WW2lep.exactly2lep.root")

t = f.Get("mini")

for n in t.GetListOfBranches(): print(n)

for i, e in enumerate(t):
    print(e.jet_n)
    if i > 10: break
