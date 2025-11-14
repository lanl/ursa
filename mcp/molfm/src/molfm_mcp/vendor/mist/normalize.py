from typing import Any, Dict, Optional

import torch
from torch.masked import MaskedTensor


class AbstractNormalizer(torch.nn.Module):
    def __init__(self, num_outputs: Optional[int] = None):
        super().__init__()
        self.num_outputs = num_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove normalization"""
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization"""
        raise NotImplementedError

    def _fit(self, x: MaskedTensor) -> dict:
        """Fit the normalization parameters"""
        raise NotImplementedError

    def to_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "num_outputs": self.num_outputs,
        }

    def fit(self, ds) -> dict:
        """Fit the normalization parameters on dataset"""
        if isinstance(ds, "IterableDataset"):
            target = []
            mask = []
            for x in ds:
                target.append(x["target"])
                mask.append(x["target_mask"])

            target = torch.stack(target)
            mask = torch.stack(mask)

        else:
            target = torch.stack([torch.tensor(x) for x in ds["target"]])
            mask = torch.stack([torch.tensor(x) for x in ds["target_mask"]])

        # Use masked tensor to compute normalization parameters
        target = MaskedTensor(target, ~mask)

        state = self._fit(target)
        return state

    @classmethod
    def get(
        cls, transform: list[str] | str | None, num_outputs: int
    ) -> "AbstractNormalizer":
        if isinstance(transform, list):
            assert len(transform) == num_outputs
            return ChannelWiseTransform([cls.get(t, 1) for t in transform])
        elif transform in ["standardize", Standardize.__name__]:
            return Standardize(num_outputs)
        elif transform in ["power_transform", PowerTransform.__name__]:
            return PowerTransform(num_outputs)
        elif transform in ["log_transform", LogTransform.__name__]:
            return LogTransform(num_outputs)
        elif transform in ["max_scale", MaxScaleTransform.__name__]:
            return MaxScaleTransform(num_outputs)
        else:
            return IdentityTransform()


class ChannelWiseTransform(AbstractNormalizer):
    def __init__(self, transforms: list[AbstractNormalizer]):
        super().__init__(len(transforms))
        self.transforms = torch.nn.ModuleList(transforms)

    def to_config(self) -> dict:
        return {
            "class": [t.__class__.__name__ for t in self.transforms],
            "num_outputs": self.num_outputs,
        }

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                transform.inverse(x[:, [idx]])
                for idx, transform in enumerate(self.transforms)
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                transform.forward(x[:, [idx]])
                for idx, transform in enumerate(self.transforms)
            ],
            dim=1,
        )

    def _fit(self, x: MaskedTensor) -> dict:
        for idx, transform in enumerate(self.transforms):
            transform._fit(x[:, [idx]])
        return self.state_dict()


class Standardize(AbstractNormalizer):
    def __init__(self, num_outputs: int, eps: float = 1e-8):
        super().__init__(num_outputs)
        self.register_buffer("mean", torch.zeros(num_outputs))
        self.register_buffer("std", torch.zeros(num_outputs))
        self.eps = float(eps)
        assert 0 <= self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.std * x) + self.mean

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _fit(self, target: MaskedTensor) -> dict:
        self.mean = target.mean(0).get_data().to(self.mean)
        self.std = target.std(0).get_data().to(self.std) + self.eps
        return self.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "transform.mean" in state_dict.keys():
            state_dict["transform.mean"] = state_dict["transform.mean"].view(1)
            state_dict["transform.std"] = state_dict["transform.std"].view(1)


class LogTransform(Standardize):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(super().forward(x))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return super().inverse(torch.log(x))

    def _fit(self, target: MaskedTensor) -> dict:
        return super()._fit(torch.log(target))


class PowerTransform(AbstractNormalizer):
    """
    Apply a power transform (Yeo-Johnson) featurewise to make data more Gaussian-like.
    Followed by applying a zero-mean, unit-variance normalization to the
    transformed output to rescale targets to [-1, 1].
    """

    def __init__(self, num_outputs, eps: float = 1e-8):
        super().__init__(num_outputs)
        self.num_outputs = num_outputs
        self.register_buffer("lmbdas", torch.zeros(num_outputs))
        self.register_buffer("mean", torch.zeros(num_outputs))
        self.register_buffer("std", torch.zeros(num_outputs))
        self.eps = float(eps)
        assert 0 <= self.eps

    def _yeo_johnson_transform(self, x, lmbda):
        """
        Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/fbb32eae5/sklearn/preprocessing/_data.py#L3354
        """
        x_out = x.clone()
        eps = torch.finfo(x.dtype).eps
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(lmbda) < eps:
            x_out[pos] = torch.log1p(x[pos])
        else:  # lmbda != 0
            x_out[pos] = (torch.pow(x[pos] + 1, lmbda) - 1) / lmbda

        # when x < 0
        if abs(lmbda - 2) > eps:
            x_out[~pos] = -(torch.pow(-x[~pos] + 1, 2 - lmbda) - 1) / (
                2 - lmbda
            )
        else:  # lmbda == 2
            x_out[~pos] = -torch.log1p(-x[~pos])

        return x_out

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """
        Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/fbb32eae5/sklearn/preprocessing/_data.py#L3383
        """
        x_out = x.clone()
        pos = x >= 0
        eps = torch.finfo(x.dtype).eps

        # when x >= 0
        if abs(lmbda) < eps:  # lmbda == 0
            x_out[pos] = torch.exp(x[pos]) - 1
        else:  # lmbda != 0
            x_out[pos] = torch.pow(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > eps:  # lmbda != 2
            x_out[~pos] = 1 - torch.pow(
                -(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda)
            )
        else:  # lmbda == 2
            x_out[~pos] = 1 - torch.exp(-x[~pos])
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Undo standardization
        x = (self.std * x) + self.mean
        x_out = torch.zeros_like(x)
        for i in range(self.num_outputs):
            x_out[:, i] = self._yeo_johnson_inverse_transform(
                x[:, i], self.lmbdas[i]
            )
        return x_out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x_out = torch.zeros_like(x)
        for i in range(self.num_outputs):
            x_out[:, i] = self._yeo_johnson_transform(x[:, i], self.lmbdas[i])
        # Standardization
        x_out = (x_out - self.mean) / self.std
        return x_out

    def _fit(self, target: MaskedTensor) -> dict:
        # Fit Yeo-Johnson lambdas
        from sklearn.preprocessing import (
            PowerTransformer as _PowerTransformer,  # noqa: F811
        )

        transformer = _PowerTransformer(method="yeo-johnson", standardize=False)
        target = torch.tensor(
            transformer.fit_transform(target.get_data().numpy())
        )
        self.lmbdas = torch.tensor(transformer.lambdas_)
        # Fit standardization scaling
        self.mean = target.mean(0).to(self.mean)
        self.std = target.std(0).to(self.std) + self.eps
        return self.state_dict()


class MaxScaleTransform(AbstractNormalizer):
    """
    Divide by maximum value in training dataset.
    """

    def __init__(self, mx: int, eps: float = 1e-8):
        super().__init__(1)
        self.num_outputs = 1
        self.max = mx
        self.eps = float(eps)
        assert 0 <= self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Undo standardization
        x_out = self.max * x
        return x_out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x / self.max
        return x_out

    def _fit(self, target: MaskedTensor) -> dict:
        return self.state_dict()


class IdentityTransform(AbstractNormalizer):
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _fit(self, x: MaskedTensor) -> dict:
        return self.state_dict()
