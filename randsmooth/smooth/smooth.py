import torch

from abc import ABC, abstractmethod


class Smooth(ABC):
    def __init__(self, base, name):
        self.base = base
        self.name = name

    def forward(self, x_batch):
        """Batch-wise predict that's compatible with module syntax."""
        predictions = []
        for x in x_batch:
            # TODO: check that for cifar10 dimensions are right
            predictions.append(self.predict(x.unsqueeze(0)))

        return torch.stack(predictions, dim=0)

    @abstractmethod
    def predict(self, x):
        """Return predicted class for x as a long tensor, tensor(-1) if abstain."""
        pass

    @abstractmethod
    def certify(self, x):
        pass
