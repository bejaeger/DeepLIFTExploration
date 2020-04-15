#!/usr/bin/env python

import ROOT
import numpy as np
import uproot
import os
import keras
from keras.models import Sequential
from keras import metrics
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from variables import *
from loaddata import *
from helper import *
import matplotlib.pyplot as plt
from plot import *
from atlasify import atlasify
from sklearn import preprocessing
import pickle

#####################################################
# Get dataset in dictionary with panda frames
cfg = readCfg("config.cfg", section="train")
dset = getDataInDict(cfg)

#####################################################
# Preprocessing  Standardize

# I/O to dump scaler
outfilepath = os.path.join(getOutputDir(), "training")
mkdir_p(outfilepath)

if cfg["standardize"]:
    print("INFO: Standardizing inputs...")
    scaler = preprocessing.StandardScaler(copy=False)
    vars = cfg["training_variables"]
    allEvents = dset["VBF"][vars].append(dset["Top"][vars])
    if "WW" in cfg["samples"]:
        allEvents = allEvents.append(dset["WW"][vars])
    scaler.fit(allEvents)
    dset["VBF"][vars] = scaler.transform(dset["VBF"][vars])
    dset["Top"][vars] = scaler.transform(dset["Top"][vars])
    if "WW" in cfg["samples"]:
        dset["WW"][vars] = scaler.transform(dset["WW"][vars])
    pickle.dump(scaler, open(os.path.join(outfilepath, "scaler.pkl"), "wb"))

#####################################################
# define training and test data
n_vars = len(cfg["training_variables"])

trainingFraction = 0.8
n_train_events_vbf = int(dset["VBF"].shape[0] * trainingFraction)
n_train_events_top = int(dset["Top"].shape[0] * trainingFraction)
if "WW" in cfg["samples"]:
    n_train_events_ww = int(dset["WW"].shape[0] * trainingFraction)

x_train_vbf = dset["VBF"][cfg["training_variables"]][0:n_train_events_vbf]
y_train_vbf = pd.Series(1, index=dset["VBF"][0:n_train_events_vbf].index)
x_train_top = dset["Top"][cfg["training_variables"]][0:n_train_events_top]
y_train_top = pd.Series(0, index=dset["Top"][0:n_train_events_top].index)

if "WW" in cfg["samples"]:
    x_train_ww = dset["WW"][cfg["training_variables"]][0:n_train_events_ww]
    y_train_ww = pd.Series(0, index=dset["WW"][0:n_train_events_ww].index)

x_train = x_train_vbf.append(x_train_top)
y_train = y_train_vbf.append(y_train_top)
if "WW" in cfg["samples"]:
    x_train = x_train.append(x_train_ww).values
    y_train = y_train.append(y_train_ww).values
else:
    x_train = x_train.values
    y_train = y_train.values

x_test_vbf = dset["VBF"][cfg["training_variables"]][n_train_events_vbf:]
y_test_vbf = pd.Series(1, index=dset["VBF"][n_train_events_vbf:].index)
x_test_top = dset["Top"][cfg["training_variables"]][n_train_events_top:]
y_test_top = pd.Series(0, index=dset["Top"][n_train_events_top:].index)
x_test = x_test_vbf.append(x_test_top).values
y_test = y_test_vbf.append(y_test_top).values

n_train_events = x_train.shape[0]
weight_vbf = 0.1 * n_train_events / n_train_events_vbf
weight_top = 0.9 * n_train_events / n_train_events_top
if "WW" in cfg["samples"]:
    weight_top = 0.5 * n_train_events / n_train_events_top
sample_weight_vbf = pd.Series(weight_vbf, index=dset["VBF"][0:n_train_events_vbf].index)
sample_weight_top = pd.Series(weight_top, index=dset["Top"][0:n_train_events_top].index)
sample_weight = sample_weight_vbf.append(sample_weight_top)
if "WW" in cfg["samples"]:
    weight_ww = 0.4 * n_train_events / n_train_events_ww
    sample_weight_ww = pd.Series(weight_top, index=dset["WW"][0:n_train_events_ww].index)
    sample_weight = sample_weight.append(sample_weight_ww).values
else:
    sample_weight = sample_weight.values
    
print("sample weight vbf: {}".format(weight_vbf))
print("sample weight top: {}".format(weight_top))

if "WW" in cfg["samples"]:
    print("sample weight WW: {}".format(weight_ww))

############################################################
# define the keras model
model = Sequential()
model.add(Dense(32, input_dim=n_vars, activation='relu', name="dense_0"))
# model.add(Dense(32, activation='relu', name="dense_1"))
model.add(Dense(16, activation='relu', name="dense_2"))
model.add(Dropout(0.01))
model.add(Dense(8, activation='relu', name="dense_3"))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid', name="output"))

# compile the keras model
# loss, logcosh
model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics = [metrics.binary_accuracy])

############################################################
# training settings
batchsize = 256
shuffle = True
# keras.optimizers.Adam(learning_rate=0.1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=10)
history = model.fit(x_train, y_train, epochs=int(cfg["nEpochs"]),
                    batch_size=batchsize,
                    validation_split=0.2, shuffle=shuffle,
                    sample_weight = sample_weight,
                    callbacks=[reduce_lr])

# evaluate the keras model
_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

############################################################
# Save model and make plots
model.save(os.path.join(outfilepath, "trained_model.h5"))
model.save_weights(os.path.join(outfilepath, "model_weights.h5"))
# save also architecture
arch = model.to_json()
with open(os.path.join(outfilepath, "model_architecture.json"), 'w') as arch_file:
    arch_file.write(arch)
        
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(outfilepath, "accuracy.png"))
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(outfilepath, "loss.png"))
plt.close()


############################################################
# Make network output plot

pred_vbf = model.predict(x_train_vbf.values, batch_size=batchsize)
pred_top = model.predict(x_train_top.values, batch_size=batchsize)
pred_vbf_test = model.predict(x_test_vbf.values, batch_size=batchsize)
pred_top_test = model.predict(x_test_top.values, batch_size=batchsize)

nbins = 40
ylabel = "Normalized Number of Events"
alpha = 0.6
normalize = True
ATLASlabel = ["Open Data", "$\sqrt{s} = 13\,\mathrm{TeV}$, $10\,\mathrm{fb}^{-1}$ \n H$\\rightarrow$WW, 2 leptons, 2 jets \n"]

processes = ["VBF", "Top"]
pred_data = [pred_vbf, pred_top]
labels = ["Signal Training Sample", "Background Training Sample"]
histtype = ["bar" , "bar"]
colors = ["red", "orange"]

for iproc, proc in enumerate(processes):
    plt.hist(pred_data[iproc], nbins,
             facecolor=colors[iproc], alpha=alpha,
             color=colors[iproc],
             range=(0, 1), density=normalize,
             # weights=weights[iproc],
             histtype=histtype[iproc],
             label=labels[iproc])

pred_data_test = [pred_vbf_test, pred_top_test] #, pred_vbf_test, pred_top_test]
labels_test = ["Signal Test Sample", "Background Test Sample"] #, pred_vbf_test, pred_top_test]
for itest, dat in enumerate(pred_data_test):
    x_test, y_test, yerr_test, norm_test = histpoints(dat, nbins,
                                                      yerr = "sqrt", normed = normalize)
    if itest == 0:
        savextest = x_test
    if itest == 1:
        x_test = savextest
    plt.plot(x_test, y_test, 'o', color=colors[itest], markersize=4, \
             label = labels_test[itest])
    plt.errorbar(x_test, y_test, fmt='none', yerr=yerr_test, ecolor=colors[itest])
    

#plt.ylim(bottom=0.00001) 
plt.xlabel("Network Output")
plt.ylabel(ylabel)
plt.yscale("log")
plt.ylim(0.001, 1000)
atlasify(ATLASlabel[0], ATLASlabel[1])
outfilename = "network_output.png"
plt.savefig(os.path.join(outfilepath, outfilename), dpi=360)

# allHistograms = getTrainingAndTestingHistograms(sampleHolder, MVAOut_train, MVAOut_test, args.noWeights)
# Z, Znew, p0, p0new = significance.getZandP0FromTraining(allHistograms["default"])
# plotHistograms(allHistograms, config, args.outputPath, date_time)
# saveHistogramsAsRootFile(allHistograms, config)
    
# # from keras.utils import plot_model
# # plot_model(model, to_file='model.png')

print("End of train.py script!")
