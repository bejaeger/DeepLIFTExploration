#!/usr/bin/env python

import ROOT
import numpy as np
import uproot
import keras
from keras.models import Sequential
from keras import metrics
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from variables import *
from loaddata import *
import matplotlib.pyplot as plt
from atlasify import atlasify
from sklearn import preprocessing

# X = dataset[:,0:8]
# y = dataset[:,8]
#####################################################
# Configuration

#nEvents = 1000000
nEvents =  -1
filepathVBF = "../../atlas-open-data/mc_345323.VBFH125_WW2lep.exactly2lep.root"
filepathWW = "../../atlas-open-data/mc_363492.llvv.exactly2lep.root"
filepathTop = "../../atlas-open-data/mc_410000.ttbar_lep.exactly2lep.root"
filepathWW = ""
samples = {"VBF":filepathVBF, "WW":filepathWW, "Top":filepathTop}

variables = ["mcWeight", "SumWeights", "XSection", "lep_pt", "jet_pt", "met_et", "lep_phi", "lep_eta", "jet_eta", "jet_phi"]
selections = [["jet_n", "==", 2]]

variables_to_build = ["leadLepPt", "subleadLepPt", "DPhill", "Mll",
                      "DEtajj", "PtTotal", "Mjj"]

training_variables = ["leadLepPt", "subleadLepPt", "DPhill", "PtTotal", "met_et", "Mll", "DEtajj", "Mjj"]

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


# define training dataa
n_vars = len(training_variables)
x_train_vbf = dset["VBF"][training_variables]
y_train_vbf = pd.Series(1, index=dset["VBF"].index)
x_train_top = dset["Top"][training_variables]
y_train_top = pd.Series(0, index=dset["Top"].index)
x_train = x_train_vbf.append(x_train_top).values
y_train = y_train_vbf.append(y_train_top).values

n_train_events = x_train.shape[0]
n_train_events_vbf = x_train_vbf.shape[0]
n_train_events_top = x_train_top.shape[0]
weight_vbf = 0.1 * n_train_events / n_train_events_vbf
weight_top = 0.9 * n_train_events / n_train_events_top
sample_weight_vbf = pd.Series(weight_vbf, index=dset["VBF"].index)
sample_weight_top = pd.Series(weight_top, index=dset["Top"].index)
sample_weight = sample_weight_vbf.append(sample_weight_top).values
print("sample weight vbf: {}".format(weight_vbf))
print("sample weight top: {}".format(weight_top))

# scaler = preprocessing.StandardScaler(copy=False)
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
        
# define the keras model
model = Sequential()
# model.add(Dense(32, input_dim=n_vars,  kernel_regularizer=regularizers.l2(0.01), activation='relu', name="dense_0"))
model.add(Dense(12, input_dim=n_vars, activation='relu', name="dense_1"))
# model.add(Dropout(0.4))
model.add(Dense(8, activation='relu', name="dense_2"))
# model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid', name="output"))

# compile the keras model
# loss, logcosh
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics = [metrics.binary_accuracy])

nEpochs = 80
batchsize = 256
#32
shuffle = True
learning_rate = 0.1
#early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

# fit the keras model on the dataset
# keras.optimizers.Adagrad(learning_rate=learning_rate, epsilon=None, decay=0.0)
#keras.optimizers.Adam(learning_rate=learning_rate)
#keras.optimizers.SGD(learning_rate=learning_rate)
history = model.fit(x_train, y_train, epochs=nEpochs,
                    batch_size=batchsize,
                    validation_split=0.2, shuffle=shuffle)
# callbacks=[early_stopping])
                    # sample_weight = sample_weight)

# evaluate the keras model
_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

model.save("trained_model.h5")
model.save_weights("model_weights.h5")
# save also architecture
arch = model.to_json()
with open("model_architecture.json", 'w') as arch_file:
    arch_file.write(arch)
        
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("accuracy.png")
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("loss.png")
plt.close()


pred_vbf = model.predict(x_train_vbf.values, batch_size=batchsize)
pred_top = model.predict(x_train_top.values, batch_size=batchsize)

nbins = 40
ylabel = "Number of Events"
alpha = 0.6
normalize = True
ATLASlabel = ["Open Data", "$\sqrt{s} = 13\,\mathrm{TeV}$, $10\,\mathrm{fb}^{-1}$ \n H$\\rightarrow$WW, 2 leptons, 2 jets \n"]

processes = ["VBF", "Top"]
pred_data = [pred_vbf, pred_top]
labels = ["Signal Process (Higgs)", "Bkg Process (Top)"]
colors = ["red", "orange"]

for iproc, proc in enumerate(processes):
    plt.hist(pred_data[iproc], nbins,
             facecolor=colors[iproc], alpha=alpha,
             range=(0, 1), density=normalize,
             # weights=weights[iproc],
             label=labels[iproc])

#plt.ylim(bottom=0.00001) 
plt.xlabel("Network Output")
plt.ylabel(ylabel)
plt.yscale("log")
atlasify(ATLASlabel[0], ATLASlabel[1])
plt.savefig("network_output.png", dpi=360)


# def getEmptyMVAHistogram(name, title = "", fullStats = True):
#     t = title
#     if not fullStats:
#         t = t + "reduced statistics, "
#     t = t.strip(" ,")
#     t += ";MVA output;Yields"
#     h = ROOT.TH1F(name, t, 40, 0, 1)
#     h.Sumw2()
#     return h

# def getTrainingAndTestingHistograms(sampleHolder, MVAOut_train, MVAOut_test, noWeights):
#     x_train, y_train, sampleWeights = sampleHolder.getInputForTraining()
#     x_test,  y_test = sampleHolder.getInputForTesting()
#     w_train = sampleHolder.getWeightsForTraining()
#     w_test  = sampleHolder.getWeightsForTesting()
#     fullStats = sampleHolder.hasFullStats()

#     histoTitle = ""
#     # Remap MVA output to 0:1
#     minValue = np.amin(np.append(MVAOut_test, MVAOut_test))
#     maxValue = np.amax(np.append(MVAOut_test, MVAOut_test))
#     if (1-(maxValue-minValue)) > 1./100:
#         printMessage("The MVA output range is [" + str(minValue) + ":" + str(maxValue) + "]. Remapping it to [0:1].")
#         histoTitle = histoTitle + "remapped MVA output, "
#         for i in range(len(MVAOut_test)):
#             MVAOut_test[i] = (MVAOut_test[i] - minValue) / (maxValue - minValue)
#         for i in range(len(MVAOut_train)):
#             MVAOut_train[i] = (MVAOut_train[i] - minValue) / (maxValue - minValue)

#     histograms = getEmptyMVAHistograms(histoTitle, fullStats)

#     fillHistogramsConditionally(histograms['train_sig'], histograms["train_bkg"], MVAOut_train, y_train, w_train)
#     fillHistogramsConditionally(histograms['test_sig'] , histograms["test_bkg"] , MVAOut_test , y_test , w_test)

#     allHistograms = {}
#     allHistograms["default"] = histograms

#     if noWeights:
#         histoTitle += "no weights, "
#         h_noWeights = emptyMVAHistograms(histoTitle, fullStats)
#         fillHistogramsConditionally(h_noWeights['train_sig'], h_noWeights["train_bkg"], MVAOut_train, y_train, 1)
#         fillHistogramsConditionally(h_noWeights['test_sig'] , h_noWeights["test_bkg"] , MVAOut_test , y_test , 1)
#         allHistograms["noWeights"] = h_noWeights
#     return allHistograms

# allHistograms = getTrainingAndTestingHistograms(sampleHolder, MVAOut_train, MVAOut_test, args.noWeights)
# Z, Znew, p0, p0new = significance.getZandP0FromTraining(allHistograms["default"])
# plotHistograms(allHistograms, config, args.outputPath, date_time)
# saveHistogramsAsRootFile(allHistograms, config)
    
# # from keras.utils import plot_model
# # plot_model(model, to_file='model.png')

print("End of train.py script!")
