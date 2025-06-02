from dataclasses import dataclass, field
from typing import Any

from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rheed_segmentation.utils import resolve_class


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> nn.Module:
        cls = resolve_class(self.name)
        return cls(**self.params)


@dataclass
class CriterionConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> loss._Loss:
        cls = resolve_class(self.name, default_module=loss)
        return cls(**self.params)


@dataclass
class OptimizerConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self, model: nn.Module) -> Optimizer:
        cls = resolve_class(self.name, default_module=optim)
        return cls(params=model.parameters(), **self.params)


@dataclass
class SchedulerConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self, optimizer: Optimizer) -> LRScheduler:
        cls = resolve_class(self.name, default_module=optim.lr_scheduler)
        return cls(optimizer=optimizer, **self.params)


@dataclass
class TrainingConfig:
    epoch: int
    batch_size: int
    model_config: ModelConfig
    criterion_config: CriterionConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig | None
    num_workers: int = 4

    def __init__(
        self,
        epoch: int,
        batch_size: int,
        model: dict[str, Any],
        criterion: dict[str, Any],
        optimizer: dict[str, Any],
        scheduler: dict[str, Any] | None = None,
    ) -> None:
        self.epoch = epoch
        self.batch_size = batch_size

        self.model_config = ModelConfig(**model)
        self.model = self.model_config.build()

        self.criterion_config = CriterionConfig(**criterion)
        self.criterion = self.criterion_config.build()

        self.optimizer_config = OptimizerConfig(**optimizer)
        self.optimizer = self.optimizer_config.build(self.model)

        if scheduler is not None:
            self.scheduler_config = SchedulerConfig(**scheduler)
            self.scheduler = self.scheduler_config.build(self.optimizer)
        else:
            self.scheduler_config = None
            self.scheduler = None
