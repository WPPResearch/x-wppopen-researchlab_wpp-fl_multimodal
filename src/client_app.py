from flwr.client import Client, ClientApp
from flwr.common import Context

from src.flowerClient import FlowerClient
from src.models import MODELS
from src.settings import settings


# Construct a FlowerClient with its own data set partition.
def get_client_fn():
    """
    Returns a Flower ClientApp instance that wraps a client initialization function.
    The inner client function is called by the VirtualClientEngine whenever a client is selected
    to participate in a federated learning round. It initializes a FlowerClient with the
    appropriate model and data partition

    :return: A ClientApp instance that constructs clients with specified configurations.
    """
    model_name = settings.model.name

    def client_fn(context: Context) -> Client:
        # Load model and data
        if model_name not in MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        partition_id = context.node_config["partition-id"]
        client_instance = FlowerClient(MODELS[model_name], partition_id).to_client()
        return client_instance

    client = ClientApp(client_fn=client_fn)
    return client
