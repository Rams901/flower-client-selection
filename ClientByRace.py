import argparse
import os
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import Sequential
from tensorflow.python.keras.activations import sigmoid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

# TO DO:
# Test everything In colab the data distribution
# Read more about the 2 chosen fairness metric and test it on colab
# i chould have the 2 distributions of data with 2 fairness metric each
# Client selection based on their fairness metric
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



acs_data = pd.read_csv('datasets/data_acs.csv')
total_races = 9
biased_percentage = 0.2

def build_model(input_size):
    model = Sequential()
    model.add(Dense(256, input_dim=input_size, activation='relu'))
    model.add(Dropout(0.15))  # Dropout layer for regularization
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, biased_race,model, x_train, x_test, y_train, y_test):
        self.biased_race=biased_race
        self.model = model
        self.client_id = client_id
        print("Biased race:", biased_race)
        self.x_train, self.x_test, self.y_train, self.y_test = prepare_data(client_id)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        y_pred = self.model.predict(self.x_test)
        y_pred = np.round(y_pred.flatten()).astype('int8')
        y_test = self.y_test.values.flatten()
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="micro")
        prec = precision_score(y_test, y_pred, average="micro")
        f1 = f1_score(y_test, y_pred, average="micro")
#winek 
        
        
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

        avg_spd = np.nanmean(spd_values)  # Use np.nanmean to handle nan values
        avg_equal_opportunity = np.nanmean(equal_opportunity_values)

        metrics = {
            'accuracy': acc,
            'rec': rec,
            'prec': prec,
            'f1': f1,
            'avg_spd': avg_spd,
            'avg_equal_opportunity': avg_equal_opportunity,
        }
        
        print("\nResults:")
        print("Biased race distribution:", race_distribution)
        print("\nBase Metrics:")
        print("Accuracy:", metrics['accuracy'])
        print("Recall:", metrics['rec'])
        print("Precision:", metrics['prec'])
        print("F1 Score:", metrics['f1'])
        print("\nFairness Metrics:")
        print("Average SPD:", metrics['avg_spd'])
        print("Average Equal Opportunity:", metrics['avg_equal_opportunity'])

        return metrics



def prepare_data(client_id):
    biased_race = client_id + 1
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
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
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
    client_id_1= 1
    x_train, x_test, y_train, y_test = prepare_data(client_id_1)
    model = build_model(x_train.shape[1])

    client_id_1 = 1  # Race-biased client 1
    client_id_2 = 2  # Race-biased client 2
    race_distribution = {client_id_1: 1, client_id_2: 2}  # Define race distribution for each client

    parser = argparse.ArgumentParser(description='A simple Python script with command-line arguments.')
    parser.add_argument('--race', '-r', default="1", help='Biased Race filter')
    args = parser.parse_args()

    client_id = int(args.race)
    biased_race = race_distribution.get(client_id, 1)  # Default to 1 if client_id is not found

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id, biased_race,model, x_train, x_test, y_train, y_test).to_client()
    )

if __name__ == "__main__":
    main()
