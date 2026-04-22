from collections import OrderedDict

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from skorch.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.masker_optimiser_components import GeneralisedJointModel_3cls_feat_drop
from src.settings import settings

# skorch callbacks
early_stopping_callback = EarlyStopping(
    monitor="valid_loss",  # Monitor validation loss
    patience=10,  # Stop if validation loss doesn't improve for 10 epochs
    lower_is_better=True,  # Lower loss is better
    threshold=0.001,  # Minimum change to be considered an improvement
    threshold_mode="rel",  # Relative change
)

lr_scheduler_callback = LRScheduler(
    policy=ReduceLROnPlateau,
    monitor="valid_loss",  # Monitor validation loss
    mode="min",  # We want to minimize the validation loss
    patience=5,  # Reduce LR if valid_loss doesn't improve for 5 epochs
    factor=0.5,  # Reduce LR by a factor of 0.5
    event_name="epoch_end",  # Call scheduler.step() at the end of each epoch
)


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: nn.Module


MODELS = {
    "GeneralisedJointModel_3cls_feat_drop": ModelConfig(
        model=GeneralisedJointModel_3cls_feat_drop(
            input_dims=settings.general.dims,
            embed_dim=settings.model.embed_dim,
            hidden_embed_dim=settings.model.hidden_embed_dim,
            hidden_dim=settings.model.hidden_dim,
            dropout_prob=settings.model.dropout_prob,
            modality_dropout_prob=settings.model.modality_dropout_prob,
        )
    )
}

param_space = {
    "classifier__batch_size": {"type": "categorical", "values": [128, 256]},
    "classifier__optimizer__lr_mlp": {"type": "loguniform", "low": 1e-4, "high": 1e-3},
    "classifier__optimizer__weight_decay_mlp": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
}


def get_weights(model):
    """Extract parameters from a model.

    Note this is specific to PyTorch. You might want to update this function if you use
    a more exotic model architecture or if you don't want to extrac all elements in
    state_dict.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    """Copy parameters onto the model.

    Note this is specific to PyTorch. You might want to update this function if you use
    a more exotic model architecture or if you don't want to replace the entire
    state_dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
