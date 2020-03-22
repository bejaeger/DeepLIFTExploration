import ROOT
import os



f = ROOT.TFile.Open("atlas-open-data/mc_345323.VBFH125_WW2lep.exactly2lep.root")

t = f.Get("mini")

print("First 10 branch names:")
for i, n in enumerate(t.GetListOfBranches()):
    print(n)
    if i > 10: break

for i, e in enumerate(t):
    print(e.jet_n)
    if i > 10: break
    
print("End!")
