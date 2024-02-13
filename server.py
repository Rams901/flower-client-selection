import threading
import random
import flwr as fl
import numpy as np
from flwr.server.client_manager import ClientManager
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy


class Criterion(ABC):
    """Abstract class which allows subclasses to implement criterion sampling."""

    @abstractmethod
    def select(self, client: ClientProxy) -> bool:

        """Decide whether a client should be eligible for sampling or not."""
        print(client.cid,client.get_properties())

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

def filtered_average_metrics(metrics, n=2):
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
    top3_metrics = sorted_metrics[:n]

    # Calculate mean averages for the top n metrics
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
from strategy import FedAvg
# Define strategy and the custom aggregation function for the evaluation metrics
strategy = FedAvg(evaluate_metrics_aggregation_fn=filtered_average_metrics)

class AdjustedClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self,) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        """Return the number of available clients."""
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients."""
        return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
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
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(client = self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than the number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]

    # FedCS Algorithm
    def fedcs(self, strategy, num_rounds=3):
        for round_num in range(num_rounds):
            # Sample clients using FedCS selection algorithm
            selected_clients = self.fedcs_client_selection()
            print(f"Round {round_num + 1}: Selected Clients - {[client.cid for client in selected_clients]}")
            # Perform federated learning round with the selected clients
            fl.server.start_round(
                client_manager=self,
                strategy=strategy,
                num_round=round_num,
                selected_clients=selected_clients,
            )

    # FedCS client selection algorithm
    def fedcs_client_selection(self):
        # Initialization
        selected_clients = []
        remaining_clients = list(self.all().values())
        total_distribution_time = 0
        elapsed_time = 0
        print("remaining_clients",remaining_clients)
        # Client Selection
        while remaining_clients:
            selected_client = max(
                remaining_clients,
                key=lambda client: total_distribution_time
                + client.get_tUL()
                + max(0, client.get_tUD() - elapsed_time),
            )
            remaining_clients.remove(selected_client)

            update_upload_time = selected_client.get_tUL() + max(
                0, selected_client.get_tUD() - elapsed_time
            )
            Θ0 = elapsed_time + update_upload_time
            total_time = (
                selected_client.get_Tcs()
                + total_distribution_time
                + Θ0
                + selected_client.get_Tagg()
            )
            print("elapsed_time + update_upload_time",Θ0 )
            if total_time < selected_client.get_Tround():
                elapsed_time = Θ0
                selected_clients.append(selected_client)
                print('selected clients list in the round')

        return selected_clients

# Start Flower server with FedCS
client_manager = AdjustedClientManager()
#fedcs_thread = threading.Thread(target=client_manager.fedcs, args=(strategy, 3))
# fedcs_thread.start()

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    client_manager=client_manager,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
