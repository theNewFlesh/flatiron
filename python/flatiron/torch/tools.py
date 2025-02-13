from typing import Any, Callable, Optional  # noqa F401
from flatiron.core.types import Filepath  # noqa: F401
import numpy as np  # noqa F401

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch
import torch.utils.data as torchdata
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


def _execute_epoch(
    epoch,              # type: int
    model,              # type: torch.nn.Module
    data_loader,        # type: torch.utils.data.DataLoader
    optimizer,          # type: torch.optim.Optimizer
    loss_func,          # type: torch.nn.Module
    device,             # type: torch.device
    metrics_func=None,  # type: Optional[Callable[..., dict[str, float]]]
    writer=None,        # type: Optional[SummaryWriter]
    mode='train',       # type: str
):
    # type: (...) -> None
    if mode == 'train':
        context = torch.enable_grad  # type: Any
        model.train()
    elif mode == 'test':
        context = torch.inference_mode
        model.eval()
    else:
        raise ValueError(f'Invalid mode: {mode}')

    metrics = []
    epoch_size = len(data_loader)
    with context():
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
            if mode == 'train':
                optimizer.zero_grad()

            y_pred = model(x)
            loss = loss_func(y_pred, y)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            # gather batch metrics
            batch_metrics = {}
            if metrics_func is not None:
                batch_metrics = metrics_func(y_pred=y_pred, y_true=y)
            batch_metrics['loss'] = loss
            metrics.append(batch_metrics)

            # write batch metrics
            if writer is not None and mode == 'train':
                batch_index = epoch * epoch_size + i
                for key, val in batch_metrics.items():
                    writer.add_scalar(f'{mode}_batch_{key}', val, batch_index)

    # write mean epoch metrics
    if writer is not None:
        epoch_metrics = pd.DataFrame(metrics) \
            .rename(lambda x: f'{mode}_epoch_{x}', axis=1) \
            .mean() \
            .to_dict()

        for key, val in epoch_metrics.items():
            writer.add_scalar(f'{mode}_epoch_{key}', val, epoch * epoch_size)


def train(
    model,           # type: torch.nn.Module
    x_train,         # type: np.ndarray
    y_train,         # type: np.ndarray
    x_test=None,     # type: np.ndarray
    y_test=None,     # type: np.ndarray
    callbacks=None,  # type: list
    batch_size=32,   # type: int
    epochs=50,       # type: int
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
    model = model.to(device)

    train_data = torchdata.DataLoader()
    test_data = torchdata.DataLoader()
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        device=device,
        metrics_func=metrics_func,
        writer=writer,
    )
    for i in tqdm(range(epochs)):
        _execute_epoch(epoch=i, mode='train', data_loader=train_data, **kwargs)
        _execute_epoch(epoch=i, mode='test', data_loader=test_data, **kwargs)
