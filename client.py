import tensorflow as tf
import flwr as fl
from imblearn.metrics import geometric_mean_score
import numpy as np
from sklearn.metrics import precision_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import f1_score

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)
loss, accuracy = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)
g_mean = geometric_mean_score(y_test, y_pred, average='macro')
sensitivity = sensitivity_score(y_test, y_pred, average='macro')
specificity = specificity_score(y_test, y_pred, average='macro')
prec = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

class main_client(fl.client.NumPyClient):
  def get_parameters(self):
    return model.get_weights()
  
  def fit(self, parameters, config):
    model.set_weights(parameters)

    return model.get_weights(), len(x_train), {}
  
  def evaluate(self, parameters, config):
    model.set_weights(parameters)

    return loss, len(x_train), {"accuracy": accuracy}, {"g-mean": g_mean}, {"sensitivity": sensitivity}, {"specificity": specificity}, {"precision": prec}, {"F1": f1}

fl.client.start_numpy_client("[::]:8080", main_client())
