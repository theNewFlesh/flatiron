from typing import Any, Callable, Optional  # noqa F401
from flatiron.core.dataset import Dataset  # noqa: F401
from flatiron.core.types import Compiled, Filepath  # noqa: F401

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import safetensors.torch as safetensors
import tqdm.notebook as tqdm
import torch
import torch.utils.data as torchdata

import flatiron.core.tools as fict
import flatiron.torch.loss as fi_torchloss
import flatiron.torch.metric as fi_torchmetric
import flatiron.torch.optimizer as fi_torchoptim


# CALLBACKS---------------------------------------------------------------------
class ModelCheckpoint:
    '''
    Class for saving PyTorch models.
    '''
    def __init__(self, filepath, save_freq='epoch', **kwargs):
        # type: (Filepath, str, Any) -> None
        '''
        Constructs ModelCheckpoint instance.

        Args:
            filepath (str or Path): Filepath pattern.
            save_freq (str, optional): Save frequency. Default: epoch.
        '''
        self._filepath = Path(filepath).as_posix()
        self.save_freq = save_freq

    def save(self, model, epoch):
        # type: (torch.nn.Module, int) -> None
        '''
        Save PyTorch model.

        Args:
            model (torch.nn.Module): Model to be saved.
            epoch (int): Current epoch.
        '''
        filepath = self._filepath.format(epoch=epoch)
        safetensors.save_model(model, filepath)


Callbacks = dict[str, SummaryWriter | ModelCheckpoint]


def get_callbacks(log_directory, checkpoint_pattern, checkpoint_params={}):
    # type: (Filepath, str, dict) -> Callbacks
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
    return dict(
        tensorboard=SummaryWriter(log_dir=log_directory),
        checkpoint=ModelCheckpoint(checkpoint_pattern, **checkpoint_params),
    )


# DATASET-----------------------------------------------------------------------
class TorchDataset(Dataset, torchdata.Dataset):
    '''
    Class for inheriting torch Dataset into flatiron Dataset.
    '''
    @staticmethod
    def monkey_patch(dataset, channels_first=True):
        # type: (Dataset, bool) -> TorchDataset
        '''
        Construct and monkey patch a new TorchDataset instance from a given
        Dataset.
        Pytorch expects images in with the shape (C, H , W) per default.

        Args:
            dataset (Dataset): Dataset.
            channels_first (bool, optional): Will convert any matrix of shape
                (H, W, C)  into (C, H, W). Default: True.

        Returns:
            TorchDataset: TorchDataset instance.
        '''
        this = TorchDataset(dataset.info)
        this._info = dataset._info
        this.data = dataset.data
        this.labels = dataset.labels
        this.label_axis = dataset.label_axis
        this._ext_regex = dataset._ext_regex
        this._calc_file_size = dataset._calc_file_size
        this._sample_gb = dataset._sample_gb
        this._channels_first = channels_first  # type: ignore
        return this

    def __getitem__(self, frame):
        # type: (int) -> list[torch.Tensor]
        '''
        Get tensor data by frame.

        Returns:
            lis[torch.Tensor]: List of Tensors.
        '''
        items = self.get_arrays(frame)

        # pytorch warns about arrays not being writable, this fixes that
        items = [x.copy() for x in items]

        # pytorch expects (C, H, W) because it sucks
        if self._channels_first:  # type: ignore
            arrays = items
            items = []
            for item in arrays:
                if item.ndim == 3:
                    item = np.transpose(item, (2, 0, 1))
                items.append(item)

        output = list(map(torch.from_numpy, items))
        return output


# COMPILE-----------------------------------------------------------------------
def pre_build(device):
    pass


def compile(
    model,      # type: Any
    optimizer,  # type: dict[str, Any]
    loss,       # type: dict[str, Any]
    metrics,    # type: list[str]
    device,     # type: str
    kwargs,     # type: dict[str, Any]
):
    # type: (...) -> dict[str, Any]
    '''
    Call `torch.compile` on given model with kwargs.

    Args:
        model (Any): Model to be compiled.
        optimizer (dict): Optimizer config for compilation.
        loss (str): Loss to be compiled.
        metrics (list[str]): Metrics function to be compiled.
        device (str): Hardware device to compile to.
        kwargs: Other params to be passed to `model.compile`.

    Returns:
        dict: Dict of compiled objects.
    '''
    return dict(
        model=torch.compile(model, **kwargs),
        optimizer=fi_torchoptim.get(optimizer, model),
        loss=fi_torchloss.get(loss),
        metrics=[fi_torchmetric.get(m) for m in metrics],
        device=device,
    )


# TRAIN-------------------------------------------------------------------------
def _execute_epoch(
    epoch,               # type: int
    model,               # type: torch.nn.Module
    data_loader,         # type: torch.utils.data.DataLoader
    optimizer,           # type: torch.optim.Optimizer
    loss_func,           # type: torch.nn.Module
    device,              # type: torch.device
    metrics_funcs=None,  # type: Optional[Callable[..., dict[str, float]]]
    writer=None,         # type: Optional[SummaryWriter]
    checkpoint=None,     # type: Optional[ModelCheckpoint]
    mode='train',        # type: str
):
    # type: (...) -> None
    '''
    Execute train or test epoch on given torch model.

    Args:
        epoch (int): Current epoch.
        model (torch.nn.Module): Torch model.
        data_loader (torch.utils.data.DataLoader): Torch data loader.
        optimizer (torch.optim.Optimizer): Torch optimizer.
        loss_func (torch.nn.Module): Torch loss function.
        device (torch.device): Torch device.
        metrics_funcs (Callable[..., dict[str, float]], optional): Torch metrics
            function. Default: None.
        writer (SummaryWriter, optional): Tensorboard writer. Default: None.
        checkpoint (ModelCheckpoint, optional): Model saver. Default: None.
        mode (str, optional): Mode to execute. Options: [train, test].
            Default: train.
    '''
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

            y_pred = model(x)
            loss = loss_func(y_pred, y)

            # train model on batch
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # gather batch metrics
            batch_metrics = {}
            if metrics_funcs is not None:
                batch_metrics = metrics_funcs(y_pred=y_pred, y_true=y)
            batch_metrics['loss'] = loss
            metrics.append(batch_metrics)

            # write batch metrics
            if writer is not None and mode == 'train':
                batch_index = epoch * epoch_size + i
                for key, val in batch_metrics.items():
                    writer.add_scalar(f'{mode}_batch_{key}', val, batch_index)

            # save model
            if checkpoint is not None and checkpoint.save_freq == 'batch':
                checkpoint.save(model, epoch)

    # write mean epoch metrics
    if writer is not None:
        epoch_metrics = pd.DataFrame(metrics) \
            .rename(lambda x: f'{mode}_epoch_{x}', axis=1) \
            .mean() \
            .to_dict()

        for key, val in epoch_metrics.items():
            writer.add_scalar(f'{mode}_epoch_{key}', val, epoch * epoch_size)


def train(
    compiled,       # type: Compiled
    callbacks,      # type: Callbacks
    train_data,     # type: Dataset
    test_data,      # type: Dataset
    batch_size=32,  # type: int
    epochs=50,      # type: int
    seed=42,        # type: int
    **kwargs,
):
    # type: (...) -> None
    '''
    Train Torch model.

    Args:
        compiled (dict): Compiled objects.
        callbacks (dict): Dict of callbacks.
        train_data (Dataset): Training dataset.
        test_data (Dataset): Test dataset.
        batch_size (int, optional): Batch size. Default: 32.
        epochs (int, optional): Number of epochs. Default: 32.
        seed (int, optional): Random seed. Default: 42.
        **kwargs: Other params to be passed to _execute_epoch.
    '''
    model = compiled['model']
    optimizer = compiled['optimizer']
    loss = compiled['loss']
    metrics = compiled['metrics']
    device = compiled['device']
    checkpoint = callbacks['checkpoint']  # type: Any

    dev = torch.device(device)
    torch.manual_seed(seed)
    model = model.to(dev)

    train_loader = torchdata.DataLoader(
        TorchDataset.monkey_patch(train_data), batch_size=batch_size
    )  # type: torchdata.DataLoader
    test_loader = torchdata.DataLoader(
        TorchDataset.monkey_patch(test_data), batch_size=batch_size
    )  # type: torchdata.DataLoader

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        loss_func=loss,
        device=dev,
        metrics_funcs=metrics,
        writer=callbacks['tensorboard'],
    )
    for i in tqdm.trange(epochs):
        _execute_epoch(
            epoch=i, mode='train', data_loader=train_loader, checkpoint=checkpoint,
            **kwargs
        )
        _execute_epoch(epoch=i, mode='test', data_loader=test_loader, **kwargs)
        if checkpoint.save_freq == 'epoch':
            checkpoint.save(model, i)
