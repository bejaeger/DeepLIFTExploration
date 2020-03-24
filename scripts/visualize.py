#!/usr/bin/env python
import os

from loaddata import loadData
from variables import *
import matplotlib.pyplot as plt
from atlasify import atlasify
from helper import *

#####################################################
# Configuration

nEvents = 10000
filepathVBF = "../../atlas-open-data/mc_345323.VBFH125_WW2lep.exactly2lep.root"
filepathWW = "../../atlas-open-data/mc_363492.llvv.exactly2lep.root"
filepathTop = "../../atlas-open-data/mc_410000.ttbar_lep.exactly2lep.root"
filepathWW = ""
samples = {"VBF":filepathVBF, "WW":filepathWW, "Top":filepathTop}

variables = ["mcWeight", "lep_pt", "jet_pt", "met_et"]
selections = [["jet_n", "==", 2]]

variables_to_build = ["leadLepPt", "subleadLepPt"]

#####################################################
# load data
dset = {}
for processName, filepath in samples.items():
    if filepath:
        print("Loading data for process '{}' from filepath '{}' " \
              "".format(processName,filepath))
        df = loadData(filepath, variables, selections = selections,
                      nEvents = nEvents)
        df = addVariables(variables_to_build, df)
        dset[processName] = df

print("==>> Branch names:")
for i, n in enumerate(dset["VBF"].keys()):
    print(n)

#######################################################
# plot settings
normalize = True
pltprocess = "VBF"
pxrange = (0, 150)
alpha = 0.6

# labels
ATLAS = "Open Data"
INFO = "$\sqrt{s} = 13\,\mathrm{TeV}$, $139\,\mathrm{fb}^{-1}$ \n" \
    "H$\\rightarrow$WW, 2 leptons, 2 jets \n"

plt.rc('axes', labelsize=12)
plt.rc('legend', fontsize=12)

plt.hist(dset[pltprocess]["leadLepPt"] / 1000., 20, facecolor='red', alpha=alpha, range=pxrange, density=normalize, weights=dset[pltprocess]["mcWeight"], label="Signal Process (Higgs)")

plt.hist(dset["Top"]["leadLepPt"] / 1000., 20, facecolor='orange', alpha=alpha, range=pxrange, density=normalize, weights=dset["Top"]["mcWeight"], label="Bkg Process (Top)")


plt.xlabel("Leading Lepton Pt [GeV]")
plt.ylabel("Normalized Number of Events", )

atlasify(ATLAS, INFO)

try:
    outputdir = os.environ["OUTPUTDIR"]
except KeyError:
    print("ERROR: Environment variable 'OUTPUTDIR' not found! " \
          "Maybe you forgot to execute 'source setup.sh'?")
    os._exit(0)
    
outfilename = "leadLepPt.png"
mkdir_p(os.path.join(outputdir, "plots"))
outfilepath = os.path.join(outputdir, "plots", outfilename)
print("Saving figure: {}".format(outfilepath))
plt.savefig(outfilepath, dpi=360)

# plt.show()
# plt.savefig("leadLepPt.pdf")


print("End!")
