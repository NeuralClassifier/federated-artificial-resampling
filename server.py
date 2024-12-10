import flwr as fl
import sys
import tensorflow as tf


numberOfclients = int(float(str(sys.argv[1])))

# Start Flower server for three rounds of federated learning

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # Use the last 5k training examples as a validation set
    x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, accuracy

    return evaluate

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(128, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#if __name__ == "__main__":
    #strategy = fl.server.strategy.FedAvg(
        #fraction_fit=0.1,
        #min_available_clients=numberOfclients)
#fl.server.start_server("[::]:8080",config={"num_rounds": 3},strategy=strategy)

strategy = fl.server.strategy.FedAvg(
    # ... other FedAvg arguments
    eval_fn=get_eval_fn(model),
)

fl.server.start_server("[::]:8080", config={"num_rounds": 3}, strategy=strategy)

