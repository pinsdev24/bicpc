import torch
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class PolynomialLRClippingConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    force_anneal: Optional[int] = field(
        default=None,
        metadata={"help": "force annealing at specified epoch"},
    )
    end_learning_rate: float = field(
        default=0.0,
        metadata={"help": "learning rate to decay to"},
    )
    power: float = field(
        default=2.0,
        metadata={"help": "decay exponent"},
    )
    total_num_update: float = field(
        default=II("optimization.max_update"),
        metadata={"help": "total number of updates over which to decay learning rate"},
    )
    lr: List[float] = II("optimization.lr")
    max_grad_norm: float = field(
        default=5.0,
        metadata={"help": "max norm of the gradients"},
    )


@register_lr_scheduler("polynomial_lr_clipping", dataclass=PolynomialLRClippingConfig)
class PolynomialLRClippingScheduler(FairseqLRScheduler):
    """Polynomial learning rate scheduler with power 2 and gradient clipping with maximum norm 5.0."""

    def __init__(self, cfg: PolynomialLRClippingConfig, optimizer):
        super().__init__(cfg, optimizer)

        assert cfg.total_num_update > 0

        self.lr = cfg.lr[0]
        if cfg.warmup_updates > 0:
            self.warmup_factor = 1.0 / cfg.warmup_updates
        else:
            self.warmup_factor = 1
        self.end_learning_rate = cfg.end_learning_rate
        self.total_num_update = cfg.total_num_update
        self.power = cfg.power
        self.max_grad_norm = cfg.max_grad_norm
        self.optimizer.set_lr(self.warmup_factor * self.lr)

    def get_next_lr(self, epoch):
        lrs = self.cfg.lr
        if self.cfg.force_anneal is None or epoch < self.cfg.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # anneal based on lr_shrink
            next_lr = self.optimizer.get_lr()
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()
    
    def step_update(self, num_updates):
        """Update the learning rate after each update and apply gradient clipping."""
        if self.cfg.warmup_updates > 0 and num_updates <= self.cfg.warmup_updates:
            self.warmup_factor = num_updates / float(self.cfg.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif num_updates >= self.total_num_update:
            lr = self.end_learning_rate
        else:
            warmup = self.cfg.warmup_updates
            lr_range = self.lr - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_update - warmup)
            lr = lr_range * pct_remaining ** self.power + self.end_learning_rate
        self.optimizer.set_lr(lr)

        # Gradient clipping
        max_norm = self.cfg.max_grad_norm
        if max_norm is not None and max_norm > 0:
            params = self.optimizer.get_model().parameters()
            torch.nn.utils.clip_grad_norm_(params, max_norm)

        return self.optimizer.get_lr()