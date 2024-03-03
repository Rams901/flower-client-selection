import argparse
import os
from typing import Dict
import flwr as fl
from flwr.common import Config
from flwr.common.context import Context
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import Sequential
from tensorflow.python.keras.activations import sigmoid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import random
import uuid
import fcntl
import time

# TO DO:
# Test everything In colab the data distribution
# Read more about the 2 chosen fairness metric and test it on colab
# i chould have the 2 distributions of data with 2 fairness metric each
# Client selection based on their fairness metric

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

acs_data = pd.read_csv('datasets/data_acs.csv')
acs_data.drop(columns = ['Unnamed: 0'], inplace = True)

# Save everything in this dataset
# epoch?
# spd? y_pred?

clients_data = pd.read_csv('datasets/clients.csv', )
total_races = 9
biased_percentage = 0.3

def build_model(input_size):

    model = Sequential()
    model.add(Dense(512, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.3))  # Increased dropout for regularization
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id, biased_race,):
        self.biased_race=biased_race
        self.cid = client_id

        super().__init__()        
        
        super().set_context({'cid': self.cid})
        print("Biased race:", biased_race)

        self.x_train, self.x_test, self.y_train, self.y_test = prepare_data(client_id)

        self.model = build_model(self.x_train.shape[1])

    def get_properties(self, config: Dict[str, bool | bytes | float | int | str]) -> Dict[str, bool | bytes | float | int | str]:
        return {'cid': self.cid, 'race': self.biased_race}
    
    def set_context(self, context: Context) -> None:
        return super().set_context({'cid': self.cid, 'race': self.biased_race})
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("CONFIG", self.context)
        print(self.get_properties({}))
        self.model.set_weights(parameters)

        self.model.fit(self.x_train, self.y_train, epochs=50, batch_size=64)

        return self.model.get_weights(), len(self.x_train), self.context

    def evaluate(self, parameters, config):

        print("CONTEXT EVAL",self.get_context(), self.context, )
        self.model.set_weights(parameters)
        y_pred = self.model.predict(self.x_test)
        print(f"Y pred: {np.sum(y_pred >= 0.5)}\nY true: {np.sum(self.y_test ==1)}")
        print(f'y pred MinMax: {np.min(y_pred), np.max(y_pred)}')

        print(f'x test: {self.x_test}')

        y_pred = np.round(y_pred.flatten()).astype('int8')
        print(np.sum(y_pred == 1))
        y_test = self.y_test.values.flatten()
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="micro")
        prec = precision_score(y_test, y_pred, average="micro")
        f1 = f1_score(y_test, y_pred, average="micro")        
        
        # Fairness metrics
        spd_values = []
        equal_opportunity_values = []
        race_distribution = create_biased_distribution(self.biased_race)

        for race in race_distribution:
            race_samples = (self.x_test['RAC1P'] == race).values
            race_predictions = y_pred[race_samples]
            other_predictions = y_pred[~race_samples]

            spd = np.abs(np.mean(race_predictions) - np.mean(other_predictions))
            spd_values.append(spd)
            # Calculate true positive rate (recall)
            true_positives = np.sum((y_pred[race_samples] == 1) & (y_test[race_samples] == 1))
            actual_positives = np.sum(y_test[race_samples] == 1)
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            equal_opportunity_values.append(recall)

        print(spd_values)

        avg_spd = np.nanmean(spd_values)  # Use np.nanmean to handle nan values
        avg_equal_opportunity = np.nanmean(equal_opportunity_values)

        metrics = {
            'acc': acc,
            'accuracy': accuracy,
            'rec': rec,
            'prec': prec,
            'f1': f1,
            'avg_spd': avg_spd,
            'avg_equal_opportunity': avg_equal_opportunity,
        }

        row = metrics
        row['client_id'] = self.cid
        row['race'] = self.biased_race
        
        #clients_data = pd.read_csv('datasets/clients.csv', )
        update = True
        while update:
            try:

                update_csv([row], "datasets/clients.csv")
                update = False

            except Exception as e:
                
                print(str(e))
                time.sleep(3)
                
        # clients_data = clients_data.to_dict('records') + [row]
        # clients_data = pd.DataFrame(clients_data)
        # clients_data.to_csv('datasets/clients.csv', index = False)

        print("\nResults:")
        print("Biased race distribution:", race_distribution)
        print("\nBase Metrics:")
        print("Accuracy:", metrics['acc'])
        print("Recall:", metrics['rec'])
        print("Precision:", metrics['prec'])
        print("F1 Score:", metrics['f1'])
        print("\nFairness Metrics:")
        print("Average SPD:", metrics['avg_spd'])
        print("Average Equal Opportunity:", metrics['avg_equal_opportunity'])

        return loss, len(self.x_test), metrics
    

def update_csv(data, csv_file):
    with open(csv_file, 'a') as file:
        
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Try to acquire the lock

        df = pd.DataFrame(data)
        df.to_csv(file, index=False, header=False, mode='a')  # Append mode

        fcntl.flock(file, fcntl.LOCK_UN)  # Release the lock

def prepare_data(biased_race):

    # biased_race = client_id + 1
    race_distribution = create_biased_distribution(biased_race)
    biased_data_sample, other_races_data = pd.DataFrame(), pd.DataFrame()

    for race, percentage in race_distribution.items():
        temp_data = acs_data[acs_data['RAC1P'] == race].sample(frac=percentage)
        if race == biased_race:
            biased_data_sample = temp_data
        else:
            other_races_data = pd.concat([other_races_data, temp_data])

    other_races_data = other_races_data[other_races_data['RAC1P'] != biased_race]
    client_data = pd.concat([biased_data_sample, other_races_data]).sample(frac=1)
    X = client_data.drop(columns="ESR")
    y = client_data['ESR'].astype("int8")
    print(f"Y_1: {np.sum(y == 1)}, Y: {len(y)}")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

# WHY IS Y PRED ALWAYS THE SAME VALUE?
    #     'biased_race': biased_race,
    #     'race_distribution': race_distribution,
    #     'biased_data_sample': biased_data_sample,  # Adjust this based on your data structure
    #     'other_races_data': other_races_data,  # Adjust this based on your data structure
    #     'client_data':client_data,
    #     'x_train': x_train,
    #     'x_test' : x_test,
    #     'y_train':y_train,
    #     'y_test':y_test
    # }

def create_biased_distribution(biased_race):
    races = list(range(1, total_races + 1)) 
    unbiased_percentage = (1.0 - biased_percentage) / (total_races - 1)
    race_distribution = {race: biased_percentage if race == biased_race else unbiased_percentage for race in races}
 
    return race_distribution

def main():

    # client_id_1 = 1  # Race-biased client 1
    # client_id_2 = 2  # Race-biased client 2
    # It's best to just init a random client with random biased distribution

    race_distribution = [race for race in range(1, 10)]  # Define race distribution for each client

    parser = argparse.ArgumentParser(description='A simple Python script with command-line arguments.')
    parser.add_argument('--race', '-r', default="-1", help='Biased Race filter')
    args = parser.parse_args()
    race = int(args.race)

    client_id = str(uuid.uuid1())
    if race == -1:
        biased_race = random.choice(race_distribution)
    else:
        biased_race = race_distribution[race - 1]

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id, biased_race,).to_client()
    )

if __name__ == "__main__":
    main()
