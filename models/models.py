from abc import ABC
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
from torch import flatten
from torch.nn import Module, Conv2d, Dropout, Linear, BatchNorm2d, ReLU, Sequential, MaxPool2d
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class AbstractModule(Module, ABC):  # TODO check that it's abstract
    def __init__(self):
        super().__init__()

        self._optim = None
        self._criterion = None
        self._scheduler = None
        self._pruner = None

    def optimizer(self, optim: callable(Optimizer), **kwargs):
        self._optim = optim(self.parameters(), **kwargs)
        return self

    def scheduler(self, scheduler: callable(LRScheduler), **kwargs):
        self._scheduler = scheduler(self._optim, **kwargs)
        return self

    def criterion(self, criterion: Module):
        self._criterion = criterion
        return self


    def fit(self,
            dataloader: DataLoader,
            epochs: int,
            callbacks=None
            ) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.to(device).train()

        for epoch in range(1, epochs + 1):
            loader_bar = tqdm(dataloader, desc='train', leave=False)
            for inputs, targets in loader_bar:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Reset the gradients (from the last iteration)
                self._optim.zero_grad()

                outputs = model(inputs)
                loss = self._criterion(outputs, targets)

                loss.backward()
                self._optim.step()

                if callbacks is not None:
                    for callback in callbacks:
                        callback()

                loader_bar.set_description(f"Epoch [{epoch}/{epochs}]")

            if self._scheduler is not None:
                self._scheduler.step()

    @torch.inference_mode()
    def evaluate(self,
                 dataloader: DataLoader,
                 verbose=True,
                 ) -> float:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.to(device).eval()

        num_samples = 0
        num_correct = 0

        for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = outputs.argmax(dim=1)

            # Update metrics
            num_samples += targets.size(0)
            num_correct += (outputs == targets).sum()

        return (num_correct / num_samples * 100).item()


class BaseLineNet(AbstractModule):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 3, 1)  # 1 x 32 x 3 x 3 = 288 parameters
        self.conv2 = Conv2d(32, 64, 3, 1)  # 32 x 64 x 3 x 3=18,432 parameters
        self.dropout1 = Dropout(0.25)
        self.dropout2 = Dropout(0.5)
        self.fc1 = Linear(9216, 128)  # 9216 x 128 = 1,179,648 parameters
        self.fc2 = Linear(128, 10)  # 128 x 10 = 1,280 parameters

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class VGG(AbstractModule):
    ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                # conv-bn-relu
                add("conv", Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", BatchNorm2d(x))
                add("relu", ReLU(True))
                in_channels = x
            else:
                add("pool", MaxPool2d(2))

        self.backbone = Sequential(OrderedDict(layers))
        self.classifier = Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
        x = self.backbone(x)

        # avgpool: [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])

        # classifier: [N, 512] => [N, 10]
        x = self.classifier(x)
        return x
