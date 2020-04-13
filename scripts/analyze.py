#!/usr/bin/env python

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
import uproot
import numpy as np
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from random import gauss
from random import random
from helper import *
from loaddata import readCfg, loadData
from variables import addVariables
import os, sys

def normalize_data(data):
    """Normalize the data so each array has a sum of 1
    Args:
      input matrix of data to be normalized
    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    norm = sum(data)
    return [1/norm * x for x in data]

#####################################################
# Read configuration and load data

cfg = readCfg("config.cfg", section="analyze")
dset = {}
for processName, filepath in cfg["samplePaths"].items():
    if filepath:
        print("Loading data for process '{}' from filepath '{}' " \
              "".format(processName,filepath))
        df = loadData(filepath, cfg["variables"], selections = cfg["selections"],
                      nEvents = cfg["nEvents"])
        df = addVariables(df, cfg["variables_to_build"])
        dset[processName] = df

# nEvents = 10000
# input_treenames = ["HWW_ttbar", "HWW_WW", "HWW_Zjets"]
# input_filepath="~/cernbox/HWW/HWWMVA/ntuples/mva_ntuple_mc_bVeto_EMPFlowJets_191217.root"

###################################################
# Settings/Definitions
minimum_DNN_cut = 0.6
cut_on_DNN = False
useMedian = False
model_input_path = "goodModel/trained_model.h5"

find_scores_layer_idx = 0
target_layer_idx=-2
scaler_filepath = "scaler.pkl"

n_vars = len(cfg["training_variables"])    

###########################
# main
deeplift_model = kc.convert_model_from_saved_files(model_input_path, nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
deeplift_contribs_func = deeplift_model.get_target_contribs_func( \
                                                                  find_scores_layer_idx=find_scores_layer_idx, \
                                                                  target_layer_idx=target_layer_idx)

#######################################
# get inputs
mean_scores, std_scores = [], []
for processName, filepath in cfg["samplePaths"].items():
    # inputs = np.array([(val) for (key,val) in X.items()]).transpose()
    # x_train_top = dset["Top"]
    inputs = np.array(dset[processName][cfg["training_variables"]])
    
    # how to include DNNoutputG in inputs!? save them during training?
    # get them from feeding through model

    #############################
    # Cut on DNN??
    
    # X_with_DNN = tree.arrays(branches=branches_with_DNN, entrystop=nEvents, namedecode = "ascii")
    # inputs_with_DNN = np.array([(val) for (key,val) in X_with_DNN.items()]).transpose()
    # inputs_with_DNN = inputs_with_DNN[np.where(inputs_with_DNN[:, (len(branches_with_DNN)-1)] > minimum_DNN_cut), :]
    # # the previous line changes the shape, adjust the shape again
    # inputs_with_DNN = inputs_with_DNN.reshape(inputs_with_DNN.shape[1], inputs_with_DNN.shape[2])
    # # remove DNN output
    # if cut_on_DNN: inputs = inputs_with_DNN[:, 0:(len(branches_with_DNN)-1)]
    
    ###############################################################

    print("==>> Processing inputs with treename '{}' and shape {}" \
          "".format(processName, inputs.shape))
    
    # scaler = preprocessing.StandardScaler(copy=False)
    # scaler.fit(inputs[:, 0:n_vars])
    # scaler = pickle.load(open(scaler_filepath, "rb"))
    # scaler.transform(inputs[:, 0:n_vars])

    # smeared_inputs = np.array([np.array([(gauss(0,1))*entry for entry in inp]) for inp in inputs])
    # smeared_inputs = np.array([np.array([(2*random()-1)*entry for entry in inp]) for inp in inputs])
    # smeared_inputs = np.array([np.array([(2*random()-1) for entry in inp]) for inp in inputs])
    # smeared_inputs1 = np.array([np.array([(2*random()-1) for entry in inp]) for inp in inputs[0:2]])
    # smeared_inputs2 = np.array([np.array([entry for entry in inp]) for inp in inputs[2:]])
    # smeared_inputs = np.concatenate((smeared_inputs1, smeared_inputs2))

    
    smeared_inputs = np.array([np.array([0 for entry in inp]) for inp in inputs])
    # scaler.transform(smeared_inputs[:, 0:n_vars])
    scores = np.array(deeplift_contribs_func(task_idx=0,  
                                             input_data_list=[inputs],
                                             input_references_list=[smeared_inputs],
                                             batch_size=100,  
                                             progress_update=1000))
    
    mean_score = scores.mean(0)
    if useMedian:
        mean_score = np.median(scores, axis=0)

    sorting_idz = np.flip(np.array(mean_score).argsort())
    for i in range(n_vars):
        # print("{} = {}".format(branches[sorting_idz[i]], mean_score[sorting_idz[i]]))
        print("{} = {}".format(cfg["training_variables"][i], mean_score[i]))

    # for visualization as bar charts:
    minimum_score = min(mean_score)
    if minimum_score < 0:
        mean_score = np.array([m + abs(minimum_score) + 1/10.*abs(max([m2+minimum_score for m2 in mean_score])) for m in mean_score])
    mean_scores.append(normalize_data(mean_score))
    std_scores.append(scores.std(0))

print("==>> Done with loop")

########################################
# Visualization

# for i, color in enumerate(['or', 'vb', '^g']):
    # plt.plot(branches, mean_scores[i], color)

x = np.asarray([i for i in range(n_vars)])
width = 0.6

if len(cfg["samples"]) == 3:
    pos = [-width/3, 0, width/3]
if len(cfg["samples"]) == 2:
    pos = [-width/2, 0, width/2]
elif len(cfg["samples"]) == 4:
    pos = [-width/2, -width/4, width/4, width/2]
else:
    print("ERROR: Please specify array of positions")
    sys.exit()
branches_for_plot = ["centr." if b == "sumOfCentralitiesL" else b for b in cfg["training_variables"]]

bars = []    
fig, ax = plt.subplots()

# use first tree for sorting
sorting_idz = np.flip(np.array(mean_scores[0]).argsort())
branch_names_sorted = np.array(branches_for_plot)[sorting_idz]
for i, lname in enumerate(cfg["samples"]):
    mean_scores_sorted = np.array(mean_scores[i])[sorting_idz]
    bars.append(ax.bar(x + pos[i], mean_scores_sorted, width/3, tick_label=branch_names_sorted))
                       # , yerr=std_scores[i]))
    
plt.ylabel("Mean(DeepLIFT Importance Score)")
if useMedian:
    plt.ylabel("Median(DeepLIFT Importance Score)")
    
plt.xlabel("Input Variable")

plt.xticks(rotation=45)

plt.legend(cfg["samples"])

if cut_on_DNN:
    plt.annotate('DNN > {:.2f}'.format(minimum_DNN_cut), (n_vars-n_vars / 2, max(mean_scores[0])))

outfilename = "scores.png"
outfilepath = os.path.join(getOutputDir(), "scores")
mkdir_p(outfilepath)
plt.savefig(os.path.join(outfilepath, outfilename))


# TODO
# - Barchart plot with error bars
# - option to look at specifc MVA output region!
     # --> scores might be dominated by background enriched region
# - normalize importance scores!

