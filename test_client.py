    #    print("\nResults:")
    #     print("Biased race distribution:", race_distribution)
    #     print("\nBase Metrics:")
    #     print("Accuracy:", metrics['acc'])
    #     print("Recall:", metrics['rec'])
    #     print("Precision:", metrics['prec'])
    #     print("F1 Score:", metrics['f1'])
    #     print("\nFairness Metrics:")
    #     print("Average SPD:", metrics['avg_spd'])
    #     print("Average Equal Opportunity:", metrics['avg_equal_opportunity'])
import pandas as pd
import fcntl

def update_csv(data, csv_file):
    with open(csv_file, 'a') as file:
        
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Try to acquire the lock

        df = pd.DataFrame(data)
        df.to_csv(file, index=False, header=False, mode='a')  # Append mode

        fcntl.flock(file, fcntl.LOCK_UN)  # Release the lock

columns = ['client_id', 'acc','accuracy', 'rec', 'prec', 'f1', 'avg_spd', 'avg_equal_opportunity']
# columns = ['client_id', 'Accuracy', 'Recall', 'Precision', 'F1', 'Fairness Metrics', 'spd', 'Average Equal Opportunity']
# df = pd.DataFrame([[0 for i in columns]], columns = columns)
# df.to_csv('datasets/clients.csv', index = False)
update_csv([{column: 0 for column in columns}], "datasets/clients.csv")
df = pd.read_csv('datasets/clients.csv')

print(df.head(5))
# print(df.groupby('ESR').count())
