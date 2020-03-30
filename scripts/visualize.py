#!/usr/bin/env python

from loaddata import *
from variables import *
import plot as myplt

#####################################################
# Configuration

nEvents = 10000
# -1
filepathVBF = "../../atlas-open-data/mc_345323.VBFH125_WW2lep.exactly2lep.root"
filepathWW = "../../atlas-open-data/mc_363492.llvv.exactly2lep.root"
filepathTop = "../../atlas-open-data/mc_410000.ttbar_lep.exactly2lep.root"
filepathWW = ""
samples = {"VBF":filepathVBF, "WW":filepathWW, "Top":filepathTop}

variables = ["mcWeight", "SumWeights", "XSection", "lep_pt", "jet_pt", "met_et", "lep_phi", "lep_eta", "jet_eta", "jet_phi"]
selections = [["jet_n", "==", 2]]

variables_to_build = ["leadLepPt", "subleadLepPt", "DPhill", "Mll",
                      "DEtajj", "PtTotal", "Mjj"]
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

######################################################
# plot settings
processes = ["VBF", "Top"]
labels = ["Signal Process (Higgs)", "Bkg Process (Top)"]
colors = ["red", "orange"]

# TODO, Fix normalization!
ifb = 10 # 10 inverse fb
# factor 1000 cause xsec is given in pb
norm = [(sum(dset[proc]["mcWeight"][0:nEvents]) / dset[proc]["SumWeights"] * 1000 * dset[proc]["XSection"] * dset[proc]["mcWeight"] * ifb) for proc in processes]

variables = ["Mll", "Mjj", "DPhill", "leadLepPt", "DEtajj", "PtTotal",
             "met_et"]
xlabels = ["Invariant Mass of Leptons [GeV]",
           "Invariant Mass of Jets [GeV]",
           "$\Delta \phi$ between Leptons ",
           "Leading Lepton Pt [GeV]",
           "$\Delta \eta$ between Jets",
           "Summed Pt of Objects [GeV]",
           "Missing Transverse Energy [GeV]"]
pxranges = [(0, 175), (0, 1000), (0, 3.2), (0, 175), (0, 5.5), (0, 600), (0, 250)]
nbins = [35, 30, 32, 35, 25, 30, 25]
scale = [0.001, 0.001, 1, 0.001, 1, 0.001, 0.001]

ylabel = "Number of Events"
alpha = 0.6
normalize = True

ATLASlabel = ["Open Data", "$\sqrt{s} = 13\,\mathrm{TeV}$, $10\,\mathrm{fb}^{-1}$ \n H$\\rightarrow$WW, 2 leptons, 2 jets \n"]

outputplotsdir = "plots" if nEvents > 0 else "plots-full-stats"
#######################################################


myplt.plotHists(dset, variables, labels, pxranges, processes,
                colors,  xlabels, scale, nbins, ylabel, weights=norm,
                outputplotsdir = outputplotsdir, 
                alpha=alpha, normalize=normalize,
                ATLASlabel=ATLASlabel)

print("End of visulalize script!")
