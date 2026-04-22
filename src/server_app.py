from typing import Dict

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.models import MODELS, ModelConfig, get_weights, set_weights
from src.settings import settings
from src.strategies.mean_strategy import MeanStrategy
from src.task import load_synthetic_data
from src.utils import test


def gen_evaluate_fn(model_config: ModelConfig):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        # Default values
        loss = 1.0
        metrics = {"centralized_accuracy": 100.0}
        if settings.server.server_dataset_included:
            X_test, y_test, _, _ = load_synthetic_data()
            model = model_config.model
            set_weights(model, parameters_ndarrays)
            report, loss = test(model, X_test, y_test)
            metrics = {
                "centralized_accuracy": report["accuracy"],
                "centralized_precision": report["weighted avg"]["precision"],
                "centralized_recall": report["weighted avg"]["recall"],
                "centralized_F1-score": report["weighted avg"]["f1-score"],
            }
        return loss, metrics

    return evaluate


def on_fit_config(server_round: int):
    """
    Construct `config` that clients receive when running `fit()`
    :param server_round: server round
    """
    # Activate on configurable server round
    pass
    return {}


# Define metric aggregation function
def weighted_average(metrics) -> Dict[str, float]:
    """
    Calculate the federated evaluation accuracy based on the sum of weighted accuracies
    from each client divided by the sum of all examples.
    :param metrics: List of client metrics to calculate average across all clients.
    :return: Dictionary of federated evaluation accuracy
    """
    # Multiply accuracy, precision, recall, F1-scores of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    F1_scores = [num_examples * m["F1-score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "federated_evaluate_accuracy": sum(accuracies) / sum(examples),
        "federated_precision": sum(precisions) / sum(examples),
        "federated_recall": sum(recalls) / sum(examples),
        "federated_F1-score": sum(F1_scores) / sum(examples),
    }


def get_server_fn():
    def server_fn(context: Context):
        # Read from config
        model_name = settings.model.name
        if model_name not in MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        model_config = MODELS[model_name]

        strategy_args = {
            "fraction_fit": settings.server.fraction_fit,
            "fraction_evaluate": settings.server.fraction_eval,
            "min_fit_clients": settings.client.num_clients,
            "min_evaluate_clients": settings.client.num_clients,
            "min_available_clients": settings.client.num_clients,
            "initial_parameters": ndarrays_to_parameters(get_weights(model_config.model)),
            "on_fit_config_fn": on_fit_config,
            "evaluate_fn": gen_evaluate_fn(model_config),
            "evaluate_metrics_aggregation_fn": weighted_average,
            "model_config": model_config,
        }

        # Define strategy
        match settings.server.strategy:
            case "Mean":
                strategy = MeanStrategy(**strategy_args)
            case _:
                raise ValueError(f"Invalid strategy {settings.server.strategy}")
        config = ServerConfig(num_rounds=settings.server.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)
    return server
