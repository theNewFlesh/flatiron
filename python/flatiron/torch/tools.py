from typing import Callable  # noqa F401
from flatiron.core.types import Filepath  # noqa: F401
import numpy as np  # noqa F401

from torch.utils.tensorboard import SummaryWriter
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
    writer = SummaryWriter(log_directory=log_directory)
    # callbacks = [
    #     tfcallbacks.TensorBoard(log_dir=log_directory, histogram_freq=1),
    #     tfcallbacks.ModelCheckpoint(checkpoint_pattern, **checkpoint_params),
    # ]
    return callbacks


def _train_step(
    model,        # type: torch.nn.Module
    data_loader,  # type: torch.utils.data.DataLoader
    loss_fn,      # type: torch.nn.Module
    optimizer,    # type: torch.optim.Optimizer
    metrics,      # type: Callable
    device,       # type: torch.device
):
    train_loss, train_acc = 0, 0
    model.to(device)
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += metrics(
            y_true=y,
            y_pred=y_pred.argmax(dim=1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def _test_step(
    data_loader,  # type: torch.utils.data.DataLoader
    model,        # type: torch.nn.Module
    loss_fn,      # type: torch.nn.Module
    metrics,      # type: Callable
    device,       # type: torch.device
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode(): 
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            test_pred = model(x)
            test_loss += loss_fn(test_pred, y)
            test_acc += metrics(    
                y_true=y,
                y_pred=test_pred.argmax(dim=1)
            )
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def train(
    model,           # type: torch.nn.Module
    x_train,         # type: np.ndarray
    y_train,         # type: np.ndarray
    x_test=None,     # type: np.ndarray
    y_test=None,     # type: np.ndarray
    callbacks=None,  # type: list
    batch_size=32,   # type: int
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
        **kwargs: Other params to be passed to `model.fit`.
    '''
    torch.manual_seed(42)
    n = x_train.shape[0]  # type: ignore
    val = None
    if x_test is not None and y_test is not None:
        val = (x_test, y_test)

    for epoch in tqdm(range(epochs)):
        _train_step(
            data_loader=train_dataloader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=    metrics
            )
        _test_step(data_loader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            metrics=    metrics
            )

    train_time_end_on_gpu = timer()
    total_train_time_model = print_train_time(
        start=train_time_start_on_gpu,
        end=train_time_end_on_gpu,
        device=device,
    )
