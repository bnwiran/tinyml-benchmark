import abc
import copy
import math
from abc import ABC
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import Module, BatchNorm2d, Conv2d
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Pruner(ABC):

    def __init__(self, model: Module):
        self.model = model

    @abc.abstractmethod
    def prune(self):
        pass

    @staticmethod
    @torch.no_grad()
    def sensitivity_scan(model: Module, dataloader: DataLoader, scan_step=0.1, scan_start=0.4, scan_end=1.0,
                         verbose=True):
        sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
        accuracies = []
        named_conv_weights = [(name, param) for (name, param) \
                              in model.named_parameters() if param.dim() > 1]
        for i_layer, (name, param) in enumerate(named_conv_weights):
            param_clone = param.detach().clone()
            accuracy = []
            for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
                FineGrainedPruner.__create_tensor_mask(param.detach(), sparsity=sparsity)
                acc = model.evaluate(dataloader, verbose=False)
                if verbose:
                    print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
                # restore
                param.copy_(param_clone)
                accuracy.append(acc)
            if verbose:
                print(
                    f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]',
                    end='')
            accuracies.append(accuracy)
        return sparsities, accuracies

    @staticmethod
    def plot_sensitivity_scan(model: Module, sparsities, accuracies, dense_model_accuracy):
        lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
        fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
        axes = axes.ravel()
        plot_index = 0
        for name, param in model.named_parameters():
            if param.dim() > 1:
                ax = axes[plot_index]
                ax.plot(sparsities, accuracies[plot_index])
                ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
                ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
                ax.set_ylim(80, 95)
                ax.set_title(name)
                ax.set_xlabel('sparsity')
                ax.set_ylabel('top-1 accuracy')
                ax.legend([
                    'accuracy after pruning',
                    f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
                ])
                ax.grid(axis='x')
                plot_index += 1
        fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
        fig.tight_layout()
        fig.subplots_adjust(top=0.925)
        plt.show()


class FineGrainedPruner(Pruner):

    def __init__(self, model: Module, sparsity_dict: dict):
        super().__init__(model)
        self.sparsity_dict = sparsity_dict
        self.masks = self.__create_masks()

    @torch.no_grad()
    def prune(self):
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    def __create_masks(self):
        masks = dict()
        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # we only prune conv and fc weights
                masks[name] = self.__create_tensor_mask(param, self.sparsity_dict.get(name))
        return masks

    @torch.no_grad()
    def __create_tensor_mask(self, tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
        """
        magnitude-based pruning for single tensor
        :param tensor: torch.[cuda.]Tensor, weight of conv/fc layer
        :param sparsity: float, pruning sparsity, sparsity = #zeros / #elements = 1 - #nonzeros / #elements
        :return:
            torch.[cuda.]Tensor, mask for zeros
        """
        if sparsity is None:
            sparsity = 0

        sparsity = min(max(0.0, sparsity), 1.0)
        if sparsity == 1.0:
            tensor.zero_()
            return torch.zeros_like(tensor)
        elif sparsity == 0.0:
            return torch.ones_like(tensor)

        num_elements = tensor.numel()
        num_zeros = round(num_elements * sparsity)
        magnitude = tensor.abs()
        threshold = magnitude.view(-1).kthvalue(num_zeros).values
        mask = torch.gt(magnitude, threshold)

        tensor.mul_(mask)

        return mask


class ChannelPruner(Pruner):
    def __init__(self, model: Module):
        super().__init__(model)

    @staticmethod
    def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
        """A function to calculate the number of layers to PRESERVE after pruning
        Note that preserve_rate = 1. - prune_ratio
        """

        return int(round(channels * (1 - prune_ratio)))

    @torch.no_grad()
    def channel_prune(self, prune_ratio: Union[List, float]) -> Module:
        """Apply channel pruning to each of the conv layer in the backbone
        Note that for prune_ratio, we can either provide a floating-point number,
        indicating that we use a uniform pruning rate for all layers, or a list of
        numbers to indicate per-layer pruning rate.
        """
        # sanity check of provided prune_ratio
        assert isinstance(prune_ratio, (float, list))
        n_conv = len([m for m in self.model.backbone if isinstance(m, Conv2d)])
        # note that for the ratios, it affects the previous conv output and next
        # conv input, i.e., conv0 - ratio0 - conv1 - ratio1-...
        if isinstance(prune_ratio, list):
            assert len(prune_ratio) == n_conv - 1
        else:
            prune_ratio = [prune_ratio] * (n_conv - 1)

        # we prune the convs in the backbone with a uniform ratio
        model = copy.deepcopy(self.model)  # prevent overwrite
        # we only apply pruning to the backbone features
        all_convs = [m for m in model.backbone if isinstance(m, Conv2d)]
        all_bns = [m for m in model.backbone if isinstance(m, BatchNorm2d)]
        # apply pruning. we naively keep the first k channels
        assert len(all_convs) == len(all_bns)

        for i_ratio, p_ratio in enumerate(prune_ratio):
            prev_conv = all_convs[i_ratio]
            prev_bn = all_bns[i_ratio]
            next_conv = all_convs[i_ratio + 1]
            original_channels = prev_conv.out_channels  # same as next_conv.in_channels
            n_keep = ChannelPruner.get_num_channels_to_keep(original_channels, p_ratio)

            # prune the output of the previous conv and bn
            prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
            prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
            prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
            prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
            prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

            # prune the input of the next conv (hint: just one line of code)
            next_conv.weight.set_(next_conv.weight.detach()[:, n_keep])

        return model

    # function to sort the channels from important to non-important
    @staticmethod
    def __get_input_channel_importance(weight):
        in_channels = weight.shape[1]
        importances = []
        # compute the importance for each input channel
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]
            importance = torch.norm(channel_weight)
            importances.append(importance.view(1))
        return torch.cat(importances)

    @torch.no_grad()
    def __apply_channel_sorting(self):
        model = copy.deepcopy(self.model)  # do not modify the original model
        # fetch all the conv and bn layers from the backbone
        all_convs = [m for m in model.backbone if isinstance(m, Conv2d)]
        all_bns = [m for m in model.backbone if isinstance(m, BatchNorm2d)]
        # iterate through conv layers
        for i_conv in range(len(all_convs) - 1):
            # each channel sorting index, we need to apply it to:
            # - the output dimension of the previous conv
            # - the previous BN layer
            # - the input dimension of the next conv (we compute importance here)
            prev_conv = all_convs[i_conv]
            prev_bn = all_bns[i_conv]
            next_conv = all_convs[i_conv + 1]
            # note that we always compute the importance according to input channels
            importance = self.get_input_channel_importance(next_conv.weight)
            # sorting from large to small
            sort_idx = torch.argsort(importance, descending=True)

            # apply to previous conv and its following bn
            prev_conv.weight.copy_(torch.index_select(
                prev_conv.weight.detach(), 0, sort_idx))
            for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
                tensor_to_apply = getattr(prev_bn, tensor_name)
                tensor_to_apply.copy_(
                    torch.index_select(tensor_to_apply.detach(), 0, sort_idx)
                )

            # apply to the next conv input (hint: one line of code)
            next_conv.weight.copy_(torch.index_select(
                next_conv.weight.detach(), 1, sort_idx))

        return model
