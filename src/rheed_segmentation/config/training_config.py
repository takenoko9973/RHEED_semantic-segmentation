from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rheed_segmentation.utils import resolve_class


class _BaseComponentConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(_BaseComponentConfig):
    def build(self) -> nn.Module:
        cls = resolve_class(self.name)

        if not issubclass(cls, nn.Module):
            msg = f"{self.name} is not a valid nn.Module subclass."
            raise TypeError(msg)

        return cls(**self.params)


class CriterionConfig(_BaseComponentConfig):
    def build(self) -> loss._Loss:
        cls = resolve_class(self.name, default_module=loss)

        if "weight" in self.params:
            self.params["weight"] = torch.Tensor(self.params["weight"])

        return cls(**self.params)


class OptimizerConfig(_BaseComponentConfig):
    def build(self, model: nn.Module) -> Optimizer:
        cls = resolve_class(self.name, default_module=optim)

        if not issubclass(cls, Optimizer):
            msg = f"{self.name} is not a valid Optimizer."
            raise TypeError(msg)

        return cls(params=model.parameters(), **self.params)


class SchedulerConfig(_BaseComponentConfig):
    def build(self, optimizer: Optimizer) -> LRScheduler:
        cls = resolve_class(self.name, default_module=optim.lr_scheduler)

        if not issubclass(cls, LRScheduler):
            msg = f"{self.name} is not a valid LRScheduler."
            raise TypeError(msg)

        return cls(optimizer=optimizer, **self.params)


class TrainingConfig(BaseModel):
    epoch: int = Field(..., gt=1)
    batch_size: int = Field(..., gt=1)
    train_model_config: ModelConfig = Field(alias="model")  # YAMLのキー名と合わせる
    criterion_config: CriterionConfig = Field(alias="criterion")
    optimizer_config: OptimizerConfig = Field(alias="optimizer")
    scheduler_config: SchedulerConfig | None = Field(default=None, alias="scheduler")
    num_workers: int = Field(default=4, ge=1)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model_instance"):
            self._model_instance = self.train_model_config.build()

        return self._model_instance

    @property
    def criterion(self) -> loss._Loss:
        if not hasattr(self, "_criterion_instance"):
            self._criterion_instance = self.criterion_config.build()

        return self._criterion_instance

    @property
    def optimizer(self) -> Optimizer:
        if not hasattr(self, "_optimizer_instance"):
            # model プロパティ経由でビルドされたインスタンスを利用
            self._optimizer_instance = self.optimizer_config.build(self.model)

        return self._optimizer_instance

    @property
    def scheduler(self) -> LRScheduler | None:
        if not hasattr(self, "_scheduler_instance"):
            if self.scheduler_config:
                # optimizer プロパティ経由でビルドされたインスタンスを利用
                self._scheduler_instance = self.scheduler_config.build(self.optimizer)
            else:
                self._scheduler_instance = None

        return self._scheduler_instance
