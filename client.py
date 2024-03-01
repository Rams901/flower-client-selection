# import os
# import pandas as pd
# import flwr as fl
# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from flwr_datasets import FederatedDataset
# import tensorflow as tf
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# # from tensorflow.python.keras.optimizers import Adam
# # Make TensorFlow log less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # Load data with Flower Datasets (fdsset)
# # fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
# fds=pd.read_csv('datasets\data_acs.csv')
# from sklearn.model_selection import train_test_split

# X = fds.drop(columns = "ESR")
# y = fds['ESR']

# x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# # Using Numpy format
# train_np = x_train.values
# test_np = x_test.values
# # Using Numpy format
# # train_np = train.with_format("numpy")
# # test_np = test.with_format("numpy")
# # x_train, y_train,x_test, y_test = train_test_split(fds, test_size=0.2, random_state=42)
# # x_test, y_test = test_np["img"], test_np["label"]

# # TO-DO
# # Distributing Dataset? or is it related to commenting model + Dataset
# # Where does the distribution happen? server?

# # Load model 
# model = Sequential()
# model.add(Dense(10, input_dim=train_np.shape[1], activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# # Make TensorFlow log less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# # Load model (MobileNetV2)
# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# # Load data with Flower Datasets (CIFAR-10)
# fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
# train = fds.load_full("train")
# test = fds.load_full("test")

# # Using Numpy format
# train_np = train.with_format("numpy")
# test_np = test.with_format("numpy")
# x_train, y_train = train_np["img"], train_np["label"]
# x_test, y_test = test_np["img"], test_np["label"]

# # Method for extra learning metrics calculation
# def eval_learning(y_test, y_pred):
#     acc = accuracy_score(y_test, y_pred)
#     rec = recall_score(
#         y_test, y_pred, average="micro"
#     )  # average argument required for multi-class
#     prec = precision_score(y_test, y_pred, average="micro")
#     f1 = f1_score(y_test, y_pred, average="micro")
#     return acc, rec, prec, f1

# # Define Flower client
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         model.fit(x_train, y_train, epochs=1, batch_size=32)
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test)
#         y_pred = model.predict(x_test)
#         y_pred = np.argmax(y_pred, axis=1).reshape(
#             -1, 1
#         )  # MobileNetV2 outputs 10 possible classes, argmax returns just the most probable

#         acc, rec, prec, f1 = eval_learning(y_test, y_pred)
#         output_dict = {
#             "accuracy": accuracy,  # accuracy from tensorflow model.evaluate
#             "acc": acc,
#             "rec": rec,
#             "prec": prec,
#             "f1": f1,
#         }
#         return loss, len(x_test), output_dict


# # Start Flower client
# fl.client.start_client(
#     server_address="127.0.0.1:8081", client=FlowerClient().to_client()
# )
