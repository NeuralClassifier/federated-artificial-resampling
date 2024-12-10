import argparse
from ast import operator
import os
from socketserver import ThreadingUnixStreamServer

import numpy as np
import tensorflow as tf
import pandas as pd
import flwr as fl
from tensorflow.keras.initializers import glorot_normal 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.utils import np_utils

from imblearn.over_sampling import ADASYN#
from imblearn.over_sampling import SMOTE#
from imblearn.over_sampling import KMeansSMOTE#
from imblearn.over_sampling import BorderlineSMOTE#
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import RandomOverSampler#
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek#
from imblearn.combine import SMOTEENN#

from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.metrics import classification_report_imbalanced

from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import TomekLinks
import collections
import statistics
from statistics import mean
import warnings


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        #sc = StandardScaler(with_mean=True, with_std=True)
        #self.x_train = sc.fit(self.x_train)
        #self.x_test = sc.fit(self.x_test)

    def get_parameters(self):
        """Get parameters of the local model."""
        #raise Exception("Not implemented (server-side parameter iniitialization)")
        return model.get_weights()
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        print("Starting to fit:")
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        epochs = 300

        batch_size = 300

        # Train the model using hyperparameters from config
        #print("Printing x_test and y_test:")
        #print(self.x_test)
        #print(self.y_test)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_data=(self.x_test,self.y_test),
            verbose=0,)
            #validation_split=0.3,)
        
        #print("Printing History:")
        #print(history.history)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        predict_x = self.model.predict(self.x_test) 
        y_pred = np.argmax(predict_x,axis=1)
        print()
        print("Accuracy in local model: ",accuracy)
        sns_val = sensitivity_score(self.y_test, y_pred, average='macro')
        spc_val = specificity_score(self.y_test, y_pred, average='macro')
        prec_val = precision_score(self.y_test, y_pred, average='macro')
        f1_val = f1_score(self.y_test, y_pred, average='macro')
        gmn_val = geometric_mean_score(self.y_test, y_pred, average='macro')
        print("Local SNS of the model in the Server: ", sns_val)
        print("Local SPC of the model in the Server: ", spc_val)
        print("Local Precision of the model in the Server: ", prec_val)
        print("Local F1 score of the model in the Server: ", f1_val)
        print("Local GMN of the model in the Server: ", gmn_val)
        print()
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 5), required=True)
    args = parser.parse_args()
    #resampling_list= ['Imbalanced','ClusterCentroids','CondensedNearestNeighbour','RandomUnderSampler','NeighbourhoodCleaningRule','EditedNearestNeighbours','AllKNN','RepeatedEditedNearestNeighbours','InstanceHardnessThreshold','NearMiss','OneSidedSelection','TomekLinks','ADASYN','KMeansSMOTE','SMOTE','BorderlineSMOTE-1','BorderlineSMOTE-2','SVMSMOTE','RandomOversampler','SMOTETomek','SMOTEENN']
    #resampling_list_oversample = ['Imbalanced','ADASYN','KMeansSMOTE','SMOTE','BorderlineSMOTE-1','BorderlineSMOTE-2','SVMSMOTE','RandomOversampler','SMOTETomek','SMOTEENN']
    #resampling_list = resampling_list_oversample + resampling_list_undersample
    #for resampler in resampling_list:
    # Load and compile Keras model
    #model = tf.keras.applications.EfficientNetB0(input_shape=(32, 32, 3), weights=None, classes=8)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    #model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(432,)),tf.keras.layers.Dense(128, activation = 'relu'),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(8, activation='softmax')])
    print()
    print(args)
    print()
    k_initializer = glorot_normal()
    optimize = tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,amsgrad=False,)

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-4,patience=10,verbose=0,mode="auto",baseline=None,restore_best_weights=True,)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape = (432,), activation = 'relu', kernel_initializer=k_initializer))
    #model.add(tf.keras.layers.Dense(300, activation = 'relu', kernel_initializer=k_initializer))

    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimize, metrics=["accuracy"])
    
        #model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


        # Load a subset of CIFAR-10 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(5)
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    #resampling = 'SMOTETomek'
    print()
    print('Idx as: ',idx)
    print()
    #amount = 100
    imr = 1
    #resampler = 'BorderlineSMOTE-1'
    #resampling_list_undersample= ['ClusterCentroids','CondensedNearestNeighbour','RandomUnderSampler','NeighbourhoodCleaningRule','EditedNearestNeighbours','AllKNN','RepeatedEditedNearestNeighbours','InstanceHardnessThreshold','NearMiss','OneSidedSelection','TomekLinks']
    #resampling_list_oversample = ['ADASYN','KMeansSMOTE','SMOTE','BorderlineSMOTE-1','BorderlineSMOTE-2','SVMSMOTE','RandomOversampler','SMOTETomek','SMOTEENN']
    
    #resampling_list = resampling_list_oversample + resampling_list_undersample
    #for resampler in resampling_list:    
        #train = pd.read_csv('/mnt/c/Users/kush1/OneDrive/Documents/FederatedL_Imbalance/refined_sampling_datasets/oversampling/'+resampling+'/'+str(amount)+'/emotionDataset_'+resampling+'_'+str(amount)+'_ref.csv')
    train = pd.read_csv('/Volumes/Backup Plus/WorksIn2022/FederatedL_Imbalance 3/Dataset/refined/imr_study/train_binary4_6_imr'+str(imr)+'.csv')

    train.drop(['Unnamed: 0'],axis=1,inplace=True)
    train = train.sample(frac=1)
    train.reset_index(drop=True,inplace=True)
    x_train = train.iloc[:,:-2] #change this to -2 for the original dataset and -1 for the resampled sets
    y_train = train.iloc[:,-1]
    columns_list = x_train.columns
    for i in range(len(y_train)):
        if y_train.iloc[i]==4:
            y_train.iloc[i]=0
        else:
            y_train.iloc[i]=1

    #sc = StandardScaler(with_mean=True, with_std=True)
    #sc.fit(x_train)

    # Apply the scaler to the X training data
    #x_train = sc.transform(x_train)

    
    test = pd.read_csv('/Volumes/Backup Plus/WorksIn2022/FederatedL_Imbalance 3/Dataset/refined/imr_study/test_binary4_6_imr.csv')
    test.drop(['Unnamed: 0'],axis=1,inplace=True)
    test = test.sample(frac=1)
    test.reset_index(drop=True,inplace=True)
    x_test = test.iloc[:,:-2]
    y_test = test.iloc[:,-1]
    for i in range(len(y_test)):
        if y_test.iloc[i]==4:
            y_test.iloc[i]=0
        else:
            y_test.iloc[i]=1
    # Apply the SAME scaler to the X test data
    #sc.fit(x_test)
    #x_test = sc.transform(x_test)

    #y_train = np_utils.to_categorical(y_train, 2)
    #y_test  = np_utils.to_categorical(y_test, 2)

    #return (x_train[idx * 5000 : (idx + 1) * 5000],y_train[idx * 5000 : (idx + 1) * 5000],), (x_test[idx * 15 : (idx + 1) * 15],y_test[idx * 15 : (idx + 1) * 15],) 
    division_value = int(len(x_train)/5)
    division_value_test = int(len(x_test)/5)
    return (x_train[idx * division_value : (idx + 1) * division_value],y_train[idx * division_value : (idx + 1) * division_value],), (x_test,y_test,)
    


if __name__ == "__main__":
    main()
