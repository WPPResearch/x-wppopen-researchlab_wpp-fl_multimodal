from datetime import datetime
from functools import partial

import optuna
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from skorch import NeuralNetClassifier

from src.models import ModelConfig, get_weights, param_space, set_weights
from src.settings import settings
from src.task import load_synthetic_data
from src.utils import create_pipeline, objective, test


class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(self, model_config: ModelConfig, partition_id: str):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.partition_id = partition_id
        self.model = model_config.model
        self.model.to(device)

        model_name = f"Masker_optimiser_{settings.general.text_embedding_task}_avg"
        who = "-FL_test"
        model_version = 1
        model_nickname = model_name + who + f"client_{self.partition_id}"

        # Create and run the Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GPSampler(),
            storage=f"sqlite:///../db_{self.partition_id}.sqlite3",
            study_name=model_nickname + "_v" + str(model_version) + "_" + datetime.now().strftime("%Y%m%d%H%M%S"),
            load_if_exists=True,
        )
        self.X_train, self.y_train, self.X_valid, self.y_valid = load_synthetic_data(self.partition_id)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        :param parameters : The current (global) model parameters.
        :param config : Configuration parameters which allow the server to influence training
        on the client. It can be used to communicate arbitrary values from the server to the client,
        for example, to set the number of (local) training epochs.
        """
        set_weights(self.model, parameters)
        self.study.optimize(
            partial(
                objective,
                X=self.X_train,
                y=self.y_train,
                param_space=param_space,
                client=settings.general.genai_client,
                model=self.model,
            ),
            n_trials=settings.model.optuna_n_trials,
        )

        # train and store trained model
        best_params = self.study.best_params
        best_pipeline = create_pipeline(self.model, params=best_params)
        best_pipeline.fit(self.X_train, self.y_train)

        # Extract the model (NeuralNetClassifier)
        for name, step in best_pipeline.steps:
            if isinstance(step, NeuralNetClassifier):
                self.model = step.module
                break

        # Return locally-trained model and metrics
        return (
            get_weights(self.model),
            len(self.X_train),
            {"id": self.partition_id},
        )

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.
        :param parameters : The current (global) model parameters.
        :param config : Configuration parameters which allow the server to influence evaluation
        on the client. It can be used to communicate arbitrary values from the server to the client,
        for example, to influence the number of examples used for evaluation.
        """
        set_weights(self.model, parameters)
        report, test_loss = test(self.model, self.X_valid, self.y_valid)
        return (
            test_loss,
            len(self.X_valid),
            {
                "accuracy": report["accuracy"],
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "F1-score": report["weighted avg"]["f1-score"],
            },
        )
