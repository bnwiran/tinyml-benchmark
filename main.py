import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip

from models import utils
from models.models import BaseLineNet, VGG
from prune.pruner import FineGrainedPruner

lr = 1.0
lr_step_gamma = 0.7
num_epochs = 2

model = BaseLineNet()
model.optimizer(Adadelta, lr=lr).scheduler(StepLR, step_size=1, gamma=lr_step_gamma).criterion(F.nll_loss)
# optimizer = Adadelta(model.parameters(), lr=lr)

# criterion = F.nll_loss
# scheduler = StepLR(optimizer, step_size=1, gamma=lr_step_gamma)

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])


def to_image(t: Tensor):
    return (t * 0.3081 + 0.1307).squeeze(0).to('cpu').numpy()


# image_size = 32
transforms = {
    "train": Compose([
        # RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    "test": ToTensor(),
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = MNIST(
        root="C:/AI/datasets/pytorch",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )
dataloader = {}
for split in ['train', 'test']:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=True,
    )


best_accuracy = 0
best_checkpoint = dict()


for n, p in model.named_parameters():
    print(n, type(p))

# model.fit(dataloader['train'], epochs=num_epochs)
# accuracy = model.evaluate(dataloader['test'])

# print('Model accuracy: ', accuracy)
sparsity_dict = {'conv1.weight': 1, 'conv2.weight': 1, 'fc1.weight':1}
print('Model sparsity: ', utils.get_model_sparsity(model))
pruner = FineGrainedPruner(model, sparsity_dict=sparsity_dict)
pruner.prune()
print('Model sparsity after pruning: ', utils.get_model_sparsity(model))
# print('Model size: ', utils.get_model_size(model))
# print('Model MACs: ', utils.get_model_macs(model, torch.randn(1, 1, 28, 28)))
# print('Model num. params: ', utils.get_num_parameters(model, count_nonzero_only=False))
