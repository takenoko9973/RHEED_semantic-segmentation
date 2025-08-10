from typing import Any

import torch
from pydantic import Field
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rheed_segmentation.utils import resolve_class

from .core import BaseConfig


class _BaseComponent(BaseConfig):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(_BaseComponent):
    def build(self) -> nn.Module:
        cls = resolve_class(self.name)

        if not issubclass(cls, nn.Module):
            msg = f"{self.name} is not a valid nn.Module subclass."
            raise TypeError(msg)

        return cls(**self.params)


class CriterionConfig(_BaseComponent):
    def build(self) -> loss._Loss:
        cls = resolve_class(self.name, default_module=loss)

        if "weight" in self.params:
            self.params["weight"] = torch.Tensor(self.params["weight"])
            self.params["weight"] = (
                self.params["weight"] / sum(self.params["weight"]) * len(self.params["weight"])
            )

        return cls(**self.params)


class OptimizerConfig(_BaseComponent):
    def build(self, model: nn.Module) -> Optimizer:
        cls = resolve_class(self.name, default_module=optim)

        if not issubclass(cls, Optimizer):
            msg = f"{self.name} is not a valid Optimizer."
            raise TypeError(msg)

        return cls(params=model.parameters(), **self.params)


class SchedulerConfig(_BaseComponent):
    def build(self, optimizer: Optimizer) -> LRScheduler:
        cls = resolve_class(self.name, default_module=optim.lr_scheduler)

        if not issubclass(cls, LRScheduler):
            msg = f"{self.name} is not a valid LRScheduler."
            raise TypeError(msg)

        return cls(optimizer=optimizer, **self.params)


class TrainingConfig(BaseConfig):
    epoch: int = Field(..., gt=1)
    batch_size: int = Field(..., gt=1)
    train_model_config: ModelConfig = Field(alias="model")  # YAMLのキー名と合わせる
    criterion_config: CriterionConfig = Field(alias="criterion")
    optimizer_config: OptimizerConfig = Field(alias="optimizer")
    scheduler_config: SchedulerConfig | None = Field(default=None, alias="scheduler")
    num_workers: int = Field(default=4, ge=1)

    def build_model(self) -> nn.Module:
        return self.train_model_config.build()

    def build_criterion(self) -> loss._Loss:
        return self.criterion_config.build()

    def build_optimizer(self, model: nn.Module) -> Optimizer:
        return self.optimizer_config.build(model)

    def build_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        if self.scheduler_config:
            return self.scheduler_config.build(optimizer)

        return None
