import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback

import deeplift
from deeplift.util import compile_func
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc


# function to generate Two Moons dataset
def generateTwoMoonsData(N_points=5000, noise=0.05):
    X1,y = sklearn.datasets.make_moons(n_samples=N_points, shuffle=True, noise=noise)
    X2 = np.random.normal(0,1,size=(N_points,8))
    X = np.concatenate((X1,X2),axis=1)
    return X,y

def computeDeepLiftScores(h5file, X_test):
    revealcancel_model = kc.convert_model_from_saved_files(h5_file=h5file, nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)
    revealcancel_func = revealcancel_model.get_target_contribs_func(find_scores_layer_idx=0, target_layer_idx=-2)
    scores = np.array(revealcancel_func(task_idx=0,
                    input_data_list=[X_test], 
                    input_references_list=[np.tile([0.5, 0.25, 0, 0, 0, 0, 0, 0, 0, 0], (len(X_test),1))],
                    batch_size=30,
                    progress_update=None))
    return scores

# Callback to stop NN training when target accuracy reached
class StopCallback(Callback):
    def __init__(self, acc_threshold):
        self.accuracy_threshold = acc_threshold

    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > self.accuracy_threshold):
            print("\nReached %2.2f%% accuracy, stopping training" %(self.accuracy_threshold*100))
            self.model.stop_training = True



# Generate Two Moons data
X,y = generateTwoMoonsData(N_points=5000, noise=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define simple NN with single hidden layer
model = Sequential()
model.add(Dense(30, input_dim=10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Instantiate a callback object that will stop training when accuracy reaches 99.5%
stopping_callback = StopCallback(0.995)

# Fit the model to training data
model.fit(X_train, y_train, epochs=1000, callbacks=[stopping_callback], verbose=1)

# Evaluate the model on the test data and save
filename = 'TwoMoonsModel.h5'
metrics = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test set:', metrics[1])
model.save(filename)

# Compute DeepLift scores
scores = computeDeepLiftScores(filename, X_test)
score_magnitudes = np.mean(np.abs(scores), axis=0)

# Plot the data
plt.figure()
plt.title('Two Moons, first two components')
plt.scatter(X[y==0,0], X[y==0,1], 3, c='r')
plt.scatter(X[y==1,0], X[y==1,1], 3, c='b')
plt.show()

# Plot mean importance score by feature
plt.figure()
plt.bar(range(10), score_magnitudes)
plt.title('Feature importance for Two Moons')
plt.xlabel('Feature index')
plt.ylabel('Average importance score')
plt.xticks(np.arange(10))
plt.show()

# Plot dominant importance score by test data point
idx_1 = np.where(abs(scores[:,0]) > abs(scores[:,1]))
idx_2 = np.where(abs(scores[:,0]) < abs(scores[:,1]))

plt.figure()
plt.scatter(X_test[idx_1,0], X_test[idx_1,1])
plt.scatter(X_test[idx_2,0], X_test[idx_2,1])
plt.title('Dominant importance score')
plt.legend(('X component', 'Y component'))
plt.show()