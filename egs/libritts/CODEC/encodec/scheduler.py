import math
from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler


class WarmupLrScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epoch=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.warmup_epoch = warmup_epoch
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_epoch:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ("linear", "exp")
        alpha = self.last_epoch / self.warmup_epoch
        if self.warmup == "linear":
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == "exp":
            ratio = self.warmup_ratio ** (1.0 - alpha)
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        power,
        max_epoch,
        warmup_epoch=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.power = power
        self.max_epoch = max_epoch
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_epoch, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_epoch = self.last_epoch - self.warmup_epoch
        real_max_epoch = self.max_epoch - self.warmup_epoch
        alpha = real_epoch / real_max_epoch
        ratio = (1 - alpha) ** self.power
        return ratio


class WarmupExpLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        gamma,
        interval=1,
        warmup_epoch=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.gamma = gamma
        self.interval = interval
        super(WarmupExpLrScheduler, self).__init__(
            optimizer, warmup_epoch, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_epoch = self.last_epoch - self.warmup_epoch
        ratio = self.gamma ** (real_epoch // self.interval)
        return ratio


class WarmupCosineLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        max_epoch,
        eta_ratio=0,
        warmup_epoch=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.eta_ratio = eta_ratio
        self.max_epoch = max_epoch
        super(WarmupCosineLrScheduler, self).__init__(
            optimizer, warmup_epoch, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_max_epoch = self.max_epoch - self.warmup_epoch
        return (
            self.eta_ratio
            + (1 - self.eta_ratio)
            * (1 + math.cos(math.pi * self.last_epoch / real_max_epoch))
            / 2
        )


class WarmupStepLrScheduler(WarmupLrScheduler):
    def __init__(
        self,
        optimizer,
        milestones: list,
        gamma=0.1,
        warmup_epoch=500,
        warmup_ratio=5e-4,
        warmup="exp",
        last_epoch=-1,
    ):
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupStepLrScheduler, self).__init__(
            optimizer, warmup_epoch, warmup_ratio, warmup, last_epoch
        )

    def get_main_ratio(self):
        real_epoch = self.last_epoch - self.warmup_epoch
        ratio = self.gamma ** bisect_right(self.milestones, real_epoch)
        return ratio
