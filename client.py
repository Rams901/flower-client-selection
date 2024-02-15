import os

import flwr as fl
from flwr.common.context import Context
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from flwr_datasets import FederatedDataset
import argparse

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ranim read timeout error maeneha juste mochklet cnx ama good job on upgrading pip
# installity wela?installit chnowa; akel exe te3 python 11 ey howa fech ta3Mli?
# Load model (MobileNetV2)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load data with Flower Datasets (CIFAR-10)
fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
train = fds.load_full("train")
test = fds.load_full("test")

# Using Numpy format
train_np = train.with_format("numpy")
test_np = test.with_format("numpy")
x_train, y_train = train_np["img"], train_np["label"]
x_test, y_test = test_np["img"], test_np["label"]

# Method for extra learning metrics calculation
def eval_learning(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(
        y_test, y_pred, average="micro"
    )  # average argument required for multi-class
    prec = precision_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    return acc, rec, prec, f1

# How do clients get detected in the server when registering them in the Client manager...

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def set_context(self, context: Context) -> None:
        return super().set_context(context)
    
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1).reshape(
            -1, 1
        )  # MobileNetV2 outputs 10 possible classes, argmax returns just the most probable

        acc, rec, prec, f1 = eval_learning(y_test, y_pred)
        output_dict = {
            "accuracy": accuracy,  # accuracy from tensorflow model.evaluate
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
        }
        return loss, len(x_test), output_dict

# Add args to contstruct a context
# Each client will have predefined configs such as personal information about the client
# We will later on use the criterion to equally distribute the clients demographics
# This will be later elaborated more

def main():

    parser = argparse.ArgumentParser(description='A simple Python script with command-line arguments.')

    # Add arguments
    parser.add_argument('--age', '-a', default = "42",help='Age for Client')
    parser.add_argument('--race', '-r', default = "African", help='Race for Client')
    parser.add_argument('--location', '-l', default = "Namibya", help = "Location for Client", )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    age = args.age
    race = args.race
    location = args.location
    


    client = FlowerClient()
    client.set_context({'age': age, 'race': race, 'location': location})

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080", client=client.to_client()
    )

if __name__ == "__main__":

    main()