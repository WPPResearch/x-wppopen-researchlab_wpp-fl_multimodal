from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from skorch import NeuralNetClassifier

from src.masker_optimiser_components import (
    ColumnExtractor,
    EmbeddingEncoder,
    GenAIPipeline,
    TensorTransformer,
    collate_fn_skorch_generalised_3cls,
)
from src.models import early_stopping_callback, lr_scheduler_callback
from src.settings import settings


def create_text_pipeline(column_name_pattern, skip_encoding=False, verbose=True):
    return Pipeline(
        [
            ("extract", ColumnExtractor(column_name_pattern, skip_encoding=skip_encoding)),
            ("embedding_encoder", EmbeddingEncoder(column_name_pattern, skip_encoding=skip_encoding, verbose=verbose)),
            ("transform2tensor", TensorTransformer(column_name_pattern)),
        ]
    )


def f1_for_selected_classes(y_true, y_pred, selected_classes={0: 0.3, 2: 0.7}):
    f1s = f1_score(y_true, y_pred, labels=list(selected_classes.keys()), average=None)
    return np.average(f1s, weights=list(selected_classes.values()))


def objective(trial, X, y, param_space, client, model: nn.Module):
    # Sample parameters from the given param_space dict
    params = {}
    for name, spec in param_space.items():
        if spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["values"])
        elif spec["type"] == "loguniform":
            params[name] = trial.suggest_loguniform(name, spec["low"], spec["high"])
        elif spec["type"] == "uniform":
            params[name] = trial.suggest_uniform(name, spec["low"], spec["high"])

    pipeline = create_pipeline(model, params=params)
    pipeline.setup_genai_clients(client)
    f1_selected_scorer = make_scorer(f1_for_selected_classes, greater_is_better=True)
    score = cross_val_score(pipeline, X, y, cv=3, scoring=f1_selected_scorer, verbose=True).mean()
    return score


def create_pipeline(model_instance: nn.Module, params=None, eval=False):
    """
    Creates the training pipeline.
    If trial is provided (Optuna), hyperparameters are sampled from it.
    Otherwise, defaults are used.
    """
    collate_fn_with_modalities = partial(collate_fn_skorch_generalised_3cls, modalities=settings.general.modalities)

    default_params = {
        "classifier__hidden_embed_dim": settings.model.hidden_embed_dim,
        "classifier__embed_dim": settings.model.embed_dim,
        "classifier__hidden_dim": settings.model.hidden_dim,
        "classifier__module__dropout_prob": settings.model.dropout_prob,
        "classifier__module__modality_dropout_prob": settings.model.modality_dropout_prob,
        "classifier__local_epochs": settings.model.local_epochs,
        "classifier__batch_size": settings.model.batch_size,
        "classifier__optimizer__lr": settings.model.lr,
        "classifier__optimizer__weight_decay": settings.model.weight_decay,
    }

    if params is not None:
        default_params.update(params)

    # Skorch NeuralNetClassifier
    net = NeuralNetClassifier(
        module=model_instance,
        max_epochs=default_params["classifier__local_epochs"],
        batch_size=default_params["classifier__batch_size"],
        iterator_train__collate_fn=collate_fn_with_modalities,
        iterator_valid__collate_fn=collate_fn_with_modalities,
        device=settings.general.device,
        optimizer=torch.optim.AdamW,
        optimizer__lr=default_params["classifier__optimizer__lr"],
        optimizer__weight_decay=default_params["classifier__optimizer__weight_decay"],
        criterion=torch.nn.CrossEntropyLoss().to(settings.general.device),
        verbose=2,
        callbacks=[early_stopping_callback, lr_scheduler_callback],
    )

    # Combine feature pipelines
    combined_features = FeatureUnion(
        [
            (f"{mod}_pipeline", create_text_pipeline(mod, skip_encoding=True, verbose=False))
            for mod in settings.general.modalities
        ]
    )

    if eval:
        net.initialize()
        net.module.eval()

    # Full pipeline
    pipeline = GenAIPipeline([("features", combined_features), ("classifier", net)], settings.general.model_dir)

    return pipeline


def compute_test_loss(net: NeuralNetClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame):
    net.module_.eval()

    dataset = net.get_dataset(X_test, y_test)
    iterator = net.get_iterator(dataset, training=False)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in iterator:
            # Extract raw_modalities_dict and move tensors to device
            raw_modalities_dict = X_batch["raw_modalities_dict"]
            raw_modalities_dict = {k: v.to(net.device) for k, v in raw_modalities_dict.items()}

            y_batch = y_batch.to(net.device)

            # Forward + loss
            y_pred = net.module_(raw_modalities_dict)
            loss = net.criterion_(y_pred, y_batch)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def test(model, X_test: pd.DataFrame, y_test: pd.DataFrame):
    pipeline = create_pipeline(model, eval=True)
    net: NeuralNetClassifier = pipeline.named_steps["classifier"]

    # Ensure eval mode
    net.module_.eval()
    net.train = False

    # Transform once
    features = pipeline.named_steps["features"]
    X_test_transformed = features.transform(X_test)

    # Predictions + loss
    y_pred = net.predict(X_test_transformed)
    test_loss = compute_test_loss(net, X_test_transformed, y_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    return report, test_loss
