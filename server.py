import threading
import random
import flwr as fl
import numpy as np
from flwr.server.client_manager import ClientManager

from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional
import pandas as pd
from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server import strategy

class Criterion(ABC):

    """Abstract class which allows subclasses to implement criterion sampling."""

    @abstractmethod
    def select(self, client: ClientProxy) -> bool:

        """Decide whether a client should be eligible for sampling or not."""

class GetParametersIns:
    def __init__(self, **kwargs):

        self.config = {}

class AdjustedCriterion(Criterion):

    def select(self, client: ClientProxy) -> bool:
        
        """Decide whether a client should be eligible for sampling or not.
        At init, considering that the client has never been trained we will wait until the second round"""
        clients_data = pd.read_csv('datasets/clients.csv', )

        if len(clients_data)< 1:
            return True
        
        # Global SPD
        mean_spd = clients_data['avg_spd'].mean()

        cid = client.get_properties(GetParametersIns(), timeout = 5).properties['cid']
        portion = clients_data[clients_data['client_id'] == cid]
        print(len(portion))
        print(f"CID: {cid}\nportion:{portion}\n")
        if len(portion):
            loc_spd = portion['avg_spd'].mean()
            # Comparing the client avg spd with global avg spd
            if loc_spd <= mean_spd:
                return True
            return False
        
        # If no initiation happened we will have to give the client a chance to get trained

        return True
    
# Define metrics aggregation function
# Define metrics aggregation function
# Define metrics aggregation function
    
def average_metrics(metrics):
    """Aggregate metrics from multiple clients by calculating mean averages.

    Parameters:
    - metrics (list): A list containing tuples, where each tuple represents metrics for a client.
                    Each tuple is structured as (num_examples, metric), where:
                    - num_examples (int): The number of examples used to compute the metrics.
                    - metric (dict): A dictionary containing custom metrics provided as `output_dict`
                                    in the `evaluate` method from `client.py`.

    Returns:
    A dictionary with the aggregated metrics, calculating mean averages. The keys of the
    dictionary represent different metrics, including:
    - 'accuracy': Mean accuracy calculated by TensorFlow.
    - 'acc': Mean accuracy from scikit-learn.
    - 'rec': Mean recall from scikit-learn.
    - 'prec': Mean precision from scikit-learn.
    - 'f1': Mean F1 score from scikit-learn.

    Note: If a weighted average is required, the `num_examples` parameter can be leveraged.

    Example:
        Example `metrics` list for two clients after the last round:
        [(10000, {'prec': 0.108, 'acc': 0.108, 'f1': 0.108, 'accuracy': 0.1080000028014183, 'rec': 0.108}),
        (10000, {'f1': 0.108, 'rec': 0.108, 'accuracy': 0.1080000028014183, 'prec': 0.108, 'acc': 0.108})]
    """
    # Check why the requested clients number adapt with the sample at the last iteration?
    # Here num_examples are not taken into account by using _
    accuracies_tf = np.mean([metric["accuracy"] for _, metric in metrics])
    accuracies = np.mean([metric["acc"] for _, metric in metrics])
    recalls = np.mean([metric["rec"] for _, metric in metrics])
    precisions = np.mean([metric["prec"] for _, metric in metrics])
    f1s = np.mean([metric["f1"] for _, metric in metrics])

    return {
        "accuracy": accuracies_tf,
        "acc": accuracies,
        "rec": recalls,
        "prec": precisions,
        "f1": f1s,
    }

def filtered_average_metrics(metrics, n = 2):
    """Aggregate metrics from the top 2 clients by calculating mean averages.

    Parameters:
    - metrics (list): A list containing tuples, where each tuple represents metrics for a client.
                    Each tuple is structured as (num_examples, metric), where:
                    - num_examples (int): The number of examples used to compute the metrics.
                    - metric (dict): A dictionary containing custom metrics provided as `output_dict`
                                    in the `evaluate` method from `client.py`.

    Returns:
    A dictionary with the aggregated metrics, calculating mean averages for the top 3 clients.
    The keys of the dictionary represent different metrics, including:
    - 'accuracy': Mean accuracy calculated by TensorFlow.
    - 'acc': Mean accuracy from scikit-learn.
    - 'rec': Mean recall from scikit-learn.
    - 'prec': Mean precision from scikit-learn.
    - 'f1': Mean F1 score from scikit-learn.

    Note: If a weighted average is required, the `num_examples` parameter can be leveraged.

    Example:
        Example `metrics` list for two clients after the last round:
        [(10000, {'prec': 0.108, 'acc': 0.108, 'f1': 0.108, 'accuracy': 0.1080000028014183, 'rec': 0.108}),
        (10000, {'f1': 0.108, 'rec': 0.108, 'accuracy': 0.1080000028014183, 'prec': 0.108, 'acc': 0.108})]
    """

    # Sort metrics based on some criterion, e.g., accuracy
    sorted_metrics = sorted(metrics, key=lambda x: x[1]["accuracy"], reverse=True)

    # Consider only the top 3 metrics
    top3_metrics = sorted_metrics[:2]

    # Calculate mean averages for the top 3 metrics
    accuracies_tf = np.mean([metric["accuracy"] for _, metric in top3_metrics])
    accuracies = np.mean([metric["acc"] for _, metric in top3_metrics])
    recalls = np.mean([metric["rec"] for _, metric in top3_metrics])
    precisions = np.mean([metric["prec"] for _, metric in top3_metrics])
    f1s = np.mean([metric["f1"] for _, metric in top3_metrics])

    return {
        "accuracy": accuracies_tf,
        "acc": accuracies,
        "rec": recalls,
        "prec": precisions,
        "f1": f1s,
    }

# Define strategy and the custom aggregation function for the evaluation metrics
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=filtered_average_metrics, min_available_clients= 9, fraction_fit=0.3, fraction_evaluate=0.3)

# Checkout the server configuration
class AdjustedClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        """Return the number of available clients."""
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients."""
        return len(self)
    
    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available."""
        with self._cv:
            return self._cv.wait_for(lambda: len(self.clients) >= num_clients, timeout=timeout)

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance."""
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance."""
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = AdjustedCriterion(),
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        if min_num_clients is None:
            min_num_clients = num_clients

        self.wait_for(min_num_clients)

        available_cids = list(self.clients)
        if criterion is not None:

            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]
        print(len(available_cids))
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    client_manager=AdjustedClientManager(),

    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)
