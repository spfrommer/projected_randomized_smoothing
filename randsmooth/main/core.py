from torchattacks.attack import Attack
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from randsmooth.smooth.smooth import Smooth
from randsmooth.base.base import Base
from randsmooth.utils import dirs, file_utils, pretty, torch_utils

from randsmooth.utils.torch_utils import launch_tensorboard

from typing import Type, Dict, List
from dataclasses import dataclass


@dataclass
class BaseFactory:
    name: str
    base_class: Type[Base]
    params: Dict
    epochs: int


@dataclass
class SmoothFactory:
    name: str
    base_name: str
    smooth_class: Type[Smooth]
    params: Dict


@dataclass
class AttackFactory:
    name: str
    attack_class: Type[Attack]
    params: Dict


def base_phase(base_factories: List[BaseFactory], params) -> Base:
    base_root = dirs.out_path(params.data, 'base')
    if params.train_base:
        file_utils.create_empty_directory(base_root)

    if params.tensorboard:
        launch_tensorboard(dirs.path(base_root), 6006, erase=False)

    def create_base(factory: BaseFactory):
        base_path = dirs.path(base_root, factory.name)
        base_params = {**factory.params, **{'data': params.dataset}}

        if params.train_base:
            pretty.section_print(f'Training base classifier {factory.name}')

            file_utils.create_empty_directory(base_path)

            base = factory.base_class(**base_params).to(torch_utils.device())

            logger = TensorBoardLogger(base_root, factory.name,
                                       version=0, default_hp_metric=False)

            checkpoint_dir = dirs.path(base_path, 'checkpoints')
            file_utils.create_empty_directory(checkpoint_dir)
            # checkpoint = ModelCheckpoint(checkpoint_dir, filename='model', monitor='val_loss')
            checkpoint = ModelCheckpoint(checkpoint_dir, filename='model', save_top_k=-1)

            trainer = pl.Trainer(max_epochs=factory.epochs, logger=logger,
                                 gpus=torch_utils.gpu_n(), num_sanity_val_steps=0,
                                 callbacks=[checkpoint])
            trainer.fit(base, params.dataset)
            if factory.epochs == 0:
                trainer.save_checkpoint(dirs.path(checkpoint_dir, 'model.ckpt'))
        else:
            pretty.section_print(f'Loading base classifier {factory.name}')
            checkpoint_path = dirs.path(base_path, 'checkpoints', 'model.ckpt')
            base = factory.base_class.load_from_checkpoint(checkpoint_path, **base_params)

        base = base.to(torch_utils.device())
        return base

    return {factory.name: create_base(factory) for factory in base_factories}


def smooth_phase(smooth_factories: List[SmoothFactory], bases: Dict[str, Base],
                 params) -> List[Smooth]:
    def create_smooth(factory: SmoothFactory):
        pretty.section_print(f'Smoothing {factory.base_name} with {factory.name}')

        base = bases[factory.base_name]
        return factory.smooth_class(base, factory.name, **factory.params)

    return [create_smooth(factory) for factory in smooth_factories]
