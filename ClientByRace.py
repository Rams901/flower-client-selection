import argparse
from datetime import datetime
import os
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Constants
SAMPLING_FRACTION = 0.8

fds = pd.read_csv('datasets\data_acs.csv')
# X = fds.drop(columns="ESR")
# y = fds['ESR']
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer for regularization

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def eval_learning(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average="micro")
    prec = precision_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")
    return acc, rec, prec, f1

RACE_DISTRIBUTION = {
    1.0: 0.7,  # Race 1 biased client
    2.0: 0.1,  # Race 2 biased client
    3.0: 0.2
    # Add other races with their biased percentages
}

# Specify biased races and their percentages
BIASED_RACES = [1.0, 2.0]
BIASED_PERCENTAGE = 0.7
UNBIASED_PERCENTAGE = (1.0 - BIASED_PERCENTAGE) / (len(RACE_DISTRIBUTION) - len(BIASED_RACES))

# Update the biased distribution
RACE_DISTRIBUTION = {race: BIASED_PERCENTAGE if race in BIASED_RACES else UNBIASED_PERCENTAGE for race in RACE_DISTRIBUTION}

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, race_distribution, ):
        self.client_id = client_id
        self.race_distribution = race_distribution

        # Generate biased datasets based on race distribution
        client_data = fds[fds['RAC1P'] == race_distribution[client_id]]
        client_data = client_data.sample(frac=SAMPLING_FRACTION)
        print("Client data",client_data.head(5) )
        biased_data = fds[fds['RAC1P'] != race_distribution[client_id]]
        print(biased_data.head(5))
        client_data = pd.concat([biased_data.sample(frac=(1 - SAMPLING_FRACTION)), client_data])[:1000]
        X = fds.drop(columns="ESR")
        y = fds['ESR']
        # init is overloaded with work

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.y_test = self.y_test.astype("int8")
        self.model = build_model(self.x_train.shape[1])


    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        
        # Train the model on the biased dataset
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32)
        # Why do you need the len?
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        y_pred = self.model.predict(self.x_test).astype('int8')
        spd_values = []
        equal_opportunity_values = []

        for race in RACE_DISTRIBUTION:
            race_samples = (self.x_test['RAC1P'] == race).values
            race_predictions = y_pred[race_samples]
            other_predictions = y_pred[~race_samples]

            print(f"Race {race}: Mean race prediction:", np.mean(race_predictions), "Mean other prediction:", np.mean(other_predictions))
            
            spd = np.abs(np.mean(race_predictions) - np.mean(other_predictions))
            spd_values.append(spd)

            # Calculate true positive rate (recall)
            # can you explain the true positives?
            # true positive i think when the predicted results are true by comparing it to the y_pred
            print(y_pred.shape, self.y_test.shape, race_samples, len(race_samples))
            mat = ((y_pred[race_samples] == 1) * (self.y_test[race_samples] == 1))
            print(mat, mat.shape)
            true_positives = np.sum(mat
                , )
            actual_positives = np.sum(self.y_test[race_samples] == 1)
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            equal_opportunity_values.append(recall)

        avg_spd = np.mean(spd_values)
        avg_equal_opportunity = np.mean(equal_opportunity_values)
        
        print(f"Average SPD across all races: {avg_spd}")
        print(f"Average Equal Opportunity across all races: {avg_equal_opportunity}")

        acc, rec, prec, f1 = eval_learning(self.y_test, y_pred)
        output_dict = {
            "accuracy": accuracy,
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
            "avg_spd": avg_spd,
            "avg_equal_opportunity": avg_equal_opportunity,
        }

        return loss, len(self.x_test), output_dict


def main():

    
    #model = build_model(input_size=input_size)

    client_id_1 = 1  # Race-biased client 1
    client_id_2 = 2  # Race-biased client 2
    race_distribution = {client_id_1: 1, client_id_2: 2}  # Define race distribution for each client

    parser = argparse.ArgumentParser(description='A simple Python script with command-line arguments.')
    parser.add_argument('--race', '-r', default="1", help='Biased Race filter')
    args = parser.parse_args()

    client_id = int(args.race)

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id, race_distribution,).to_client()
    )

if __name__ == "__main__":
    main()