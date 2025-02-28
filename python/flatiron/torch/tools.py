from typing import Any, Callable, Optional  # noqa F401
from flatiron.core.dataset import Dataset  # noqa: F401
from flatiron.core.types import Compiled, Filepath, Getter  # noqa F401

from copy import deepcopy
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import lunchbox.tools as lbt
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
# ------------------------------------------------------------------------------


def resolve_config(config):
    # type: (dict) -> dict
    '''
    Resolve configs handed to Torch classes. Replaces the following:

    * learning_rate
    * epsilon
    * clipping_threshold
    * exponent
    * norm_degree
    * beta_1
    * beta_2

    Args:
        config (dict): Config dict.

    Returns:
        dict: Resolved config.
    '''
    params = config.pop('params', None)
    output = deepcopy(config)
    if params is not None:
        output['params'] = params

    lut = dict(
        learning_rate='lr',
        epsilon='eps',
        clipping_threshold='d',
        exponent='p',
        norm_degree='p',
    )
    for key, val in config.items():
        if key in lut:
            output[lut[key]] = val
            del output[key]

    if 'beta_1' in output or 'beta_2' in output:
        beta_1 = output.pop('beta_1', 0.9)
        beta_2 = output.pop('beta_2', 0.999)
        output['betas'] = (beta_1, beta_2)

    return output


def get(config, module, fallback_module):
    # type: (Getter, str, str) -> Any
    '''
    Given a config and set of modules return an instance or function.

    Args:
        config (dict): Instance config.
        module (str): Always __name__.
        fallback_module (str): Fallback module, either a tf or torch module.

    Raises:
        EnforceError: If config is not a dict with a name key.

    Returns:
        object: Instance or function.
    '''
    fict.enforce_getter(config)
    # --------------------------------------------------------------------------

    config = resolve_config(config)
    name = config.pop('name')
    try:
        return fict.get_module_class(name, module)
    except NotImplementedError:
        mod = fict.get_module(fallback_module)
        return getattr(mod, name)(**config)


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
        if Path(filepath).suffix == '.safetensors':
            safetensors.save_model(model, filepath)
        else:
            torch.save(model, filepath)


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
        this._info = dataset._info.copy()
        this._info['frame'] = this._info.index
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
    framework,  # type: Getter
    model,      # type: Any
    optimizer,  # type: Getter
    loss,       # type: Getter
    metrics,    # type: list[Getter]
):
    # type: (...) -> Getter
    '''
    Call `torch.compile` on given model with kwargs.

    Args:
        framework (dict): Framework dict.
        model (Any): Model to be compiled.
        optimizer (dict): Optimizer config for compilation.
        loss (str): Loss to be compiled.
        metrics (list[str]): Metrics function to be compiled.

    Returns:
        dict: Dict of compiled objects.
    '''
    kwargs = dict(filter(
        lambda x: x[0] not in ['name', 'device'], framework.items()
    ))
    return dict(
        framework=framework,
        model=torch.compile(model, **kwargs),
        optimizer=fi_torchoptim.get(optimizer, model),
        loss=fi_torchloss.get(loss),
        metrics=[fi_torchmetric.get(m) for m in metrics],
    )


# TRAIN-------------------------------------------------------------------------
def _execute_epoch(
    epoch,             # type: int
    model,             # type: torch.nn.Module
    data_loader,       # type: torch.utils.data.DataLoader
    optimizer,         # type: torch.optim.Optimizer
    loss_func,         # type: torch.nn.Module
    device,            # type: torch.device
    metrics_funcs=[],  # type: list[Callable]
    writer=None,       # type: Optional[SummaryWriter]
    checkpoint=None,   # type: Optional[ModelCheckpoint]
    mode='train',      # type: str
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
        metrics_funcs (list[Callable], optional): List of torch metrics.
            Default: [].
        writer (SummaryWriter, optional): Tensorboard writer. Default: None.
        checkpoint (ModelCheckpoint, optional): Model saver. Default: None.
        device (torch.device): Torch device.
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
        raise ValueError(f'Invalid mode: {mode}.')

    # checkpoint mode
    checkpoint_mode = checkpoint is not None and checkpoint.save_freq == 'batch'

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
            batch_metrics = dict(loss=loss)
            for metric in metrics_funcs:
                key = lbt.to_snakecase(metric.__class__.__name__)
                batch_metrics[key] = metric(y_pred, y)
            metrics.append(batch_metrics)

            # write batch metrics
            if writer is not None:
                batch_index = epoch * epoch_size + i
                for key, val in batch_metrics.items():
                    writer.add_scalar(f'batch_{mode}_{key}', val, batch_index)

            # save model
            if checkpoint_mode:
                checkpoint.save(model, epoch)  # type: ignore

    # write mean epoch metrics
    if writer is not None:
        epoch_metrics = pd.DataFrame(metrics) \
            .map(lambda x: x.cpu().detach().numpy().mean()) \
            .rename(lambda x: f'epoch_{mode}_{x}', axis=1) \
            .mean() \
            .to_dict()

        for key, val in epoch_metrics.items():
            writer.add_scalar(key, val, epoch)


def train(
    compiled,    # type: Compiled
    callbacks,   # type: Callbacks
    train_data,  # type: Dataset
    test_data,   # type: Dataset
    params,      # type: dict
):
    # type: (...) -> None
    '''
    Train Torch model.

    Args:
        compiled (dict): Compiled objects.
        callbacks (dict): Dict of callbacks.
        train_data (Dataset): Training dataset.
        test_data (Dataset): Test dataset.
        params (dict): Training params.
    '''
    framework = compiled['framework']
    model = compiled['model']
    optimizer = compiled['optimizer']
    loss = compiled['loss']
    metrics = compiled['metrics']
    checkpoint = callbacks['checkpoint']  # type: Any
    writer = callbacks['tensorboard']
    batch_size = params['batch_size']

    device = torch.device(framework['device'])
    torch.manual_seed(params['seed'])
    model = model.to(device)
    loss = loss.to(device)
    metrics = [x.to(device) for x in metrics]

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
        device=device,
        metrics_funcs=metrics,
        writer=writer,
    )
    for i in tqdm.trange(params['epochs']):
        _execute_epoch(
            epoch=i, mode='train', data_loader=train_loader,
            checkpoint=checkpoint, **kwargs
        )
        _execute_epoch(epoch=i, mode='test', data_loader=test_loader, **kwargs)
        if checkpoint.save_freq == 'epoch':
            checkpoint.save(model, i)
