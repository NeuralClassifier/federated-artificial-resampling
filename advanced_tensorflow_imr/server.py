from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal 
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.utils import np_utils
import keras
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings

round_val = 1
resampling = ''
amount = 0
imr = 1
def main() -> None:
    warnings.filterwarnings("ignore")
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    #model = tf.keras.applications.EfficientNetB0(input_shape=(32, 32, 3), weights=None, classes=8)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    #model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(432,)),
                                    #tf.keras.layers.Dense(128, activation = 'relu'),
                                    #tf.keras.layers.Dropout(0.2),
                                    #tf.keras.layers.Dense(8, activation='softmax')])
    k_initializer = glorot_normal()
    optimize = tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,amsgrad=False,)

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=1e-4,patience=10,verbose=0,mode="auto",baseline=None,restore_best_weights=True,)
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape = (432,), activation = 'relu', kernel_initializer=k_initializer))
    #model.add(tf.keras.layers.Dense(300, activation = 'relu', kernel_initializer=k_initializer))

    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimize, metrics=["accuracy"])
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=5,
        min_eval_clients=5,
        min_available_clients=5,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 100}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # Use the last 5k training examples as a validation set
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]
    global round_val
    global resampling
    global amount
    global imr
    #resampling = 'SMOTEENN' 
    #amount = 100
    imr = 1
   
    #train = pd.read_csv('/home/kushanku/Kushankur/FederatedL_Imbalance/refined_sampling_datasets/oversampling/'+resampling+'/'+str(amount)+'/emotionDataset_'+resampling+'_'+str(amount)+'_ref.csv')
    train = pd.read_csv('/mnt/c/Users/kush1/OneDrive/Documents/FederatedL_Imbalance/Dataset/refined/imr_study/train_binary4_6_imr'+str(imr)+'.csv')
    train.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    x_train = train.iloc[:,:-2] #change this to -2 for the original dataset and -1 for the resampled sets
    y_train = train.iloc[:,-1]
    for i in range(len(y_train)):
        if y_train.iloc[i]==4:
            y_train.iloc[i]=0
        else:
            y_train.iloc[i]=1

    sc = StandardScaler(with_mean=True, with_std=True)
    sc.fit(x_train)

    # Apply the scaler to the X training data
    #x_train = sc.transform(x_train)

    test = pd.read_csv('/Volumes/Backup Plus/WorksIn2022/FederatedL_Imbalance 3/Dataset/refined/imr_study/test_binary4_6_imr.csv')
    test.drop(['Unnamed: 0'],axis=1,inplace=True)
    x_val = test.iloc[:,:-2]
    y_val = test.iloc[:,-1]

    for i in range(len(y_val)):
        if y_val.iloc[i]==4:
            y_val.iloc[i]=0
        else:
            y_val.iloc[i]=1

    #x_val = sc.transform(x_val)

    #y_train = np_utils.to_categorical(y_train, 2)
    #y_val  = np_utils.to_categorical(y_val, 2)

    #sc = StandardScaler(with_mean=True, with_std=True)
    #x_train = sc.fit(x_train)
    #x_val = sc.fit(x_val)

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        global round_val
        global reampling
        global amount
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        predict_x = model.predict(x_val) 
        y_pred = np.argmax(predict_x,axis=1)
        #y_pred = model.predict_classes(x_val)
        sns_val = sensitivity_score(y_val, y_pred, average='macro')
        spc_val = specificity_score(y_val, y_pred, average='macro')
        prec_val = precision_score(y_val, y_pred, average='macro')
        f1_val = f1_score(y_val, y_pred, average='macro')
        gmn_val = geometric_mean_score(y_val, y_pred, average='macro')

        sns_list = []
        spc_list = []
        prec_list = []
        f1_list = []
        gmn_list = []
        round_list = []
        resampling_list = []
        amount_list = []
        imr_list = []

        sns_list.append(sns_val)
        spc_list.append(spc_val)
        prec_list.append(prec_val)
        f1_list.append(f1_val)
        gmn_list.append(gmn_val)
        round_list.append(round_val)
        #resampling_list.append(resampling)
        #amount_list.append(amount)
        imr_list.append(imr)
        
        print()
        print("Round: ", round_val)
        print("Accuracy of the model in the Server: ", accuracy)
        print("SNS of the model in the Server: ", sns_val)
        print("SPC of the model in the Server: ", spc_val)
        print("Precision of the model in the Server: ", prec_val)
        print("F1 score of the model in the Server: ", f1_val)
        print("GMN of the model in the Server: ", gmn_val)
        print('Classification Report Imbalanced: ',classification_report_imbalanced(y_val,y_pred))
        print('Classification Report: ',classification_report(y_val,y_pred))
        print('Classification Report: ',confusion_matrix(y_val,y_pred))

        print()

        sns_df = pd.DataFrame(sns_list,columns=['SNS'])
        spc_df = pd.DataFrame(spc_list,columns=['SPC'])
        prec_df = pd.DataFrame(prec_list,columns=['PRE'])
        f1_df = pd.DataFrame(f1_list,columns=['F1'])
        gmn_df = pd.DataFrame(gmn_list,columns=['GMN'])
        round_df = pd.DataFrame(round_list,columns=['Rounds'])
        imr_df = pd.DataFrame(imr_list,columns=['ImR'])
        #resampling_df = pd.DataFrame(resampling_list,columns=['resampling'])
        #amount_df = pd.DataFrame(amount_list,columns=['amount'])

        final_res = pd.concat([imr_df,round_df,sns_df,spc_df,prec_df,f1_df,gmn_df],axis=1)

        #if round_val == 1 and resampling == 'BorderlineSMOTE-1' and amount == 20:
        if round_val == 1 and imr == 1:
            
            final_res.to_excel("/mnt/c/Users/kush1/OneDrive/Documents/FederatedL_Imbalance/refined_results/well_parted/imr/imr_ResFed_refined.xlsx")
            
        else:
            
            df_res = pd.read_excel("/mnt/c/Users/kush1/OneDrive/Documents/FederatedL_Imbalance/refined_results/well_parted/imr/imr_ResFed_refined.xlsx")
            existing_columns = df_res.columns
            if 'Unnamed: 0' in existing_columns:
                print(df_res)
                df_res.drop(['Unnamed: 0'],axis=1,inplace=True)
                
            final_res = pd.concat([df_res,final_res],axis=0)
            final_res.reset_index(drop=True,inplace=True)
            final_res.to_excel("/mnt/c/Users/kush1/OneDrive/Documents/FederatedL_Imbalance/refined_results/well_parted/imr/imr_ResFed_refined.xlsx")
            
        
        round_val = 1 + round_val

        
        
        return loss, {"accuracy": accuracy, "SNS": sns_val, "SPC": spc_val, "Precision": prec_val, "F1": f1_val, "GMN": gmn_val}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
