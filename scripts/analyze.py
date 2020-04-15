#!/usr/bin/env python

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
import numpy as np
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from random import gauss
from random import random
from helper import *
from loaddata import *
from variables import *
import os
from keras.models import load_model


def normalize_data(data):
    """Normalize the data so each array has a sum of 1
    Args:
      input matrix of data to be normalized
    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    norm = sum(data)
    return [1/norm * x for x in data]

########################################################
# Get dataset in dictionary with panda frames

cfg = readCfg("config.cfg", section="analyze")
dset = getDataInDict(cfg)

########################################################
# Settings/Definitions

min_NN_cut = 0.0
useMedian = False

outfilepath = os.path.join(getOutputDir())
model_input_path = os.path.join(outfilepath, cfg["modelPath"])

########################################################
# Preprocessing  Standardize

scaler_filepath = os.path.join(outfilepath, cfg["scalerPath"])
if cfg["standardize"]:
    print("INFO: Standardizing inputs...")
    scaler = pickle.load(open(scaler_filepath, "rb"))
    vars = cfg["training_variables"]
    if "filepathData" in cfg.keys():
        dset["Data"][vars] = scaler.transform(dset["Data"][vars])
    else:
        dset["VBF"][vars] = scaler.transform(dset["VBF"][vars])
        dset["Top"][vars] = scaler.transform(dset["Top"][vars])
        if "WW" in cfg["samples"]:
            dset["WW"][vars] = scaler.transform(dset["WW"][vars])

########################################################
# Deeplift initializations

find_scores_layer_idx = 0
target_layer_idx=-2
n_vars = len(cfg["training_variables"])    

# load model to deeplift
deeplift_model = kc.convert_model_from_saved_files(model_input_path, nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
deeplift_contribs_func = deeplift_model.get_target_contribs_func( \
                                                                  find_scores_layer_idx=find_scores_layer_idx, \
                                                                  target_layer_idx=target_layer_idx)

########################################################
# get inputs and calculate deeplift scores!

if "filepathData" in cfg.keys():
    inputs = dset["Data"][cfg["training_variables"]].values
else:
    inputs = dset["VBF"][cfg["training_variables"]].append(dset["Top"][cfg["training_variables"]]).values

# Cut on DNN output?
model = load_model(model_input_path)
prediction = model.predict(inputs, batch_size=256)
indizes = np.where(prediction > min_NN_cut)[0]
inputs = inputs[indizes]

########################################################
# select references and calculate score

if cfg["referenceMode"] == "allzeros":
    print("INFO: choosing reference to be all-zeros!")
    reference_inputs = np.array([np.array([0 for entry in inp]) for inp in inputs])
if cfg["referenceMode"] == "gaussian":
    print("INFO: choosing reference to be a gaussian sampling (mean=0, sigma=1)!")
    reference_inputs = np.array([np.array([gauss(0,1) for entry in inp]) for inp in inputs])
if cfg["referenceMode"] == "mean":
    print("INFO: choosing reference to be the mean!")
    means = []
    for ivar in range(inputs.shape[1]): means.append(np.mean(inputs[:, ivar]))
    reference_inputs = np.array([np.array([means[ivar] for ivar, entry in enumerate(inp)]) for inp in inputs])
    for var, mean in zip(cfg["training_variables"], means):
        print("Mean({}) = {:.3f}".format(var, mean))
if cfg["referenceMode"] == "randomsampling":
    print("INFO: choosing reference to be a random sampling from input distributions!")
    nentries = len(inputs)-1
    reference_inputs = np.array([np.array([inputs[int(nentries*random()), ivar] for ivar, entry in enumerate(inp)]) for inp in inputs])

scores = np.array(deeplift_contribs_func(task_idx=0,  
                                         input_data_list=[inputs],
                                         input_references_list=[reference_inputs],
                                         batch_size=256,  
                                         progress_update=1000))
scores = np.array(list(map(abs, scores)) )
mean_score = scores.mean(0)
if useMedian:
    mean_score = np.median(scores, axis=0)

sorting_idz = np.flip(np.array(mean_score).argsort())
for i in range(n_vars):
    print("{} = {}".format(cfg["training_variables"][i], mean_score[i]))

# for visualization as bar charts:
minimum_score = min(mean_score)
if minimum_score < 0:
    mean_score = np.array([m + abs(minimum_score) + 1/10.*abs(max([m2+minimum_score for m2 in mean_score])) for m in mean_score])


########################################################
# Visualization

x = np.asarray([i for i in range(n_vars)])
width = 0.6

# custom manipulations
branches_for_plot = cfg["training_variables"]
fig, ax = plt.subplots()
# sort scores
sorting_idz = np.flip(np.array(mean_score).argsort())
branch_names_sorted = np.array(branches_for_plot)[sorting_idz]
mean_scores_sorted = np.array(mean_score)[sorting_idz]

ax.bar(x, mean_scores_sorted, width, tick_label=branch_names_sorted)

plt.ylabel("Mean DeepLIFT Importance Score")
if useMedian:
    plt.ylabel("Median DeepLIFT Importance Score")
plt.xlabel("Input Variable")
plt.xticks(rotation=30)

if min_NN_cut > 0.0:
    plt.annotate('DNN > {:.2f}'.format(min_NN_cut), (n_vars-n_vars / 2, max(mean_score)))

plt.subplots_adjust(bottom=0.18)

outfilename = "scores-refmode-{}.pdf".format(cfg["referenceMode"])
outfilepath = os.path.join(getOutputDir(), cfg["modelPath"].split("/")[0])
mkdir_p(outfilepath)
plt.savefig(os.path.join(outfilepath, outfilename))
plt.savefig(os.path.join(outfilepath, outfilename).replace(".pdf", ".png"), dpi=360)

