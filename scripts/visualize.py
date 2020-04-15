#!/usr/bin/env python

from loaddata import *
from variables import *
import plot as myplt
from sklearn import preprocessing


#####################################################
# Get dataset in dictionary with panda frames

cfg = readCfg("config.cfg", section="visualize")
dset = getDataInDict(cfg)

print("==>> Branch names:")
for i, n in enumerate(dset["VBF"].keys()):
    print(n)
    
######################################################
# plot settings

variables = ["Mll", "Mjj", "DPhill", "leadJetPt", "subleadJetPt",
             "DEtajj", "PtTotal", "met_et"]

if cfg["standardize"]:
    scaler = preprocessing.StandardScaler(copy=False)
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    allEvents = dset["VBF"][variables].append(dset["Top"][variables])
    scaler.fit(allEvents)
    dset["VBF"][variables] = scaler.transform(dset["VBF"][variables])
    dset["Top"][variables] = scaler.transform(dset["Top"][variables])

processes = ["VBF", "Top"]
labels = ["Signal Process (Higgs)", "Bkg Process (Top)"]
colors = ["red", "orange"]

# TODO, Fix normalization!
ifb = 10 # 10 inverse fb
# factor 1000 cause xsec is given in pb
norm = [(sum(dset[proc]["mcWeight"][0:cfg["nEvents"]]) / dset[proc]["SumWeights"] * 1000 * dset[proc]["XSection"] * dset[proc]["mcWeight"] * ifb) for proc in processes]

xlabels = ["Invariant Mass of Leptons (Mll) [GeV]",
           "Invariant Mass of Jets (Mjj) [GeV]",
           "$\Delta \phi$ between Leptons (DPhill)",
           "Leading Jet Transverse Momentum (leadJetPt) [GeV]",
           "Subleading Jet Transverse Momentum (subleadJetPt) [GeV]",
           "$\Delta \eta$ between Jets (DEtajj)",
           "Summed Transverse Momentum of Objects (PtTotal) [GeV]",
           "Missing Transverse Energy (met_et) [GeV]"]
pxranges = [(0, 175), (0, 1000), (0, 3.2), (0, 250), (0, 125), (0, 5.5), (0, 600), (0, 250)]
nbins = [35, 30, 32, 25, 25, 25, 30, 25]
scale = [0.001, 0.001, 1, 0.001, 0.001, 1, 0.001, 0.001]
if cfg["standardize"]:
    xlabels = ["Standardized "+l.replace(" [GeV]", "") for l in xlabels]
    pxranges = [(-4, 4) for i in range(len(variables))]
    nbins = [30 for i in range(len(variables))]
    scale = [1 for i in range(len(variables))]

ylabel = "Number of Events"
alpha = 0.6
normalize = True

ATLASlabel = ["Open Data", "$\sqrt{s} = 13\,\mathrm{TeV}$, $10\,\mathrm{fb}^{-1}$ \n H$\\rightarrow$WW, 2 leptons, 2 jets \n"]

outputplotsdir = "plots" if cfg["nEvents"] > 0 else "plots-full-stats"
if cfg["standardize"]:
    outputplotsdir = outputplotsdir+"-standardized"


#######################################################
# plot
myplt.plotHists(dset, variables, labels, pxranges, processes,
                colors,  xlabels, scale, nbins, ylabel, weights=norm,
                outputplotsdir = outputplotsdir, 
                alpha=alpha, normalize=normalize,
                ATLASlabel=ATLASlabel)

print("End of visulalize script!")
