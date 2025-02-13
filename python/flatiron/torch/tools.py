from typing import Any, Callable, Optional  # noqa F401
from flatiron.core.types import Filepath  # noqa: F401
import numpy as np  # noqa F401

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch
import tqdm.notebook as tqdm

import flatiron.core.tools as fict
# ------------------------------------------------------------------------------


def get_callbacks(log_directory, checkpoint_pattern, checkpoint_params={}):
    # type: (Filepath, str, dict) -> list
    '''
    Create a list of callbacks for Tensoflow model.

    Args:
        log_directory (str or Path): Tensorboard project log directory.
        checkpoint_pattern (str): Filepath pattern for checkpoint callback.
        checkpoint_params (dict, optional): Params to be passed to checkpoint
            callback. Default: {}.

    Raises:
        EnforceError: If log directory does not exist.
        EnforeError: If checkpoint pattern does not contain '{epoch}'.

    Returns:
        list: Tensorboard and ModelCheckpoint callbacks.
    '''
    fict.enforce_callbacks(log_directory, checkpoint_pattern)
    callbacks = [
        SummaryWriter(log_directory=log_directory),
        # torch.save function
    ]
    return callbacks


def _train_epoch(
    epoch,              # type: int
    model,              # type: torch.nn.Module
    data_loader,        # type: torch.utils.data.DataLoader
    optimizer,          # type: torch.optim.Optimizer
    loss_func,          # type: torch.nn.Module
    device,             # type: torch.device
    metrics_func=None,  # type: Optional[Callable[Any, dict[str, float]]]
    writer=None,        # type: Optional[SummaryWriter]
):
    # type: (...) -> dict[str, float]
    model.train()

    metrics = []
    epoch_size = len(data_loader)

    model.to(device)
    for i, batch in enumerate(data_loader):
        # get x and y
        if len(batch) == 2:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
        else:
            x = batch
            x = x.to(device)
            y = x

        # train model on batch
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()

        # gather batch metrics
        batch_metrics = {}
        if metrics_func is not None:
            batch_metrics = metrics_func(y_pred=y_pred, y_true=y)
        batch_metrics['loss'] = loss
        metrics.append(batch_metrics)

        # write to tensorboard
        if writer is not None:
            batch_index = epoch * epoch_size + i
            for key, val in batch_metrics.items():
                writer.add_scalar(f'train_batch_{key}', val, batch_index)

    # compute mean epoch metrics
    output = pd.DataFrame(metrics) \
        .rename(lambda x: f'train_epoch_{x}', axis=1) \
        .mean() \
        .to_dict()
    return output
    data_loader,  # type: torch.utils.data.DataLoader
    model,        # type: torch.nn.Module
    loss_fn,      # type: torch.nn.Module
    metrics,      # type: Callable
    device,       # type: torch.device
):
    # type: (...) -> tuple[float, float]
    loss = 0
    score = 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            test_pred = model(x)
            loss += loss_fn(test_pred, y)
            score += metrics(
                y_true=y,
                y_pred=test_pred.argmax(dim=1)
            )
        loss /= len(data_loader)
        score /= len(data_loader)
    return loss, score


def train(
    model,           # type: torch.nn.Module
    x_train,         # type: np.ndarray
    y_train,         # type: np.ndarray
    x_test=None,     # type: np.ndarray
    y_test=None,     # type: np.ndarray
    callbacks=None,  # type: list
    batch_size=32,   # type: int
    seed=42,         # type: int
    **kwargs,
):
    # type: (...) -> None
    '''
    Train Torch model.

    Args:
        model (tfmodels.Model): Torch model.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray, optional): Test data. Default: None.
        y_test (np.ndarray, optional): Test labels. Default: None.
        callbacks (list, optional): List of callbacks. Default: None.
        batch_size (int, optional): Batch size. Default: 32.
        seed (int, optional): Random seed. Default: 42.
        **kwargs: Other params to be passed to `model.fit`.
    '''
    torch.manual_seed(seed)
    n = x_train.shape[0]
    val = None
    if x_test is not None and y_test is not None:
        val = (x_test, y_test)

    for epoch in tqdm(range(epochs)):
        _train_step(
            data_loader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics
            )
        _test_step(
            data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            metrics=metrics
        )
