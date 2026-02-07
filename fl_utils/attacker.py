import sys
sys.path.append("../")
import time
import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18, layer2module
import copy
import os
import math

class Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.setup()

    def setup(self):
        # 初始化手工回合数
        self.handcraft_rnds = 0

        # 选择设备：如果有CUDA可用，使用GPU；否则使用CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据数据集设置触发器和掩码
        if self.helper.config.dataset in ['cifar10', 'cifar100', 'TinyImageNet']:
            channels, height, width = 3, 32, 32
        elif self.helper.config.dataset in ['FEMNIST', 'MNIST']:
            channels, height, width = 1, 28, 28
        else:
            raise ValueError(f"Unsupported dataset: {self.helper.config.dataset}")

        # 创建触发器
        trigger_value = self.helper.config.get('trigger_value', 0.5)  # 默认触发器值为0.5
        self.trigger = torch.ones((1, channels, height, width), requires_grad=False, device=device) * trigger_value

        # 创建掩码
        self.mask = torch.zeros_like(self.trigger, device=device)
        trigger_size = self.helper.config.get('trigger_size', 5)  # 默认触发器尺寸为5
        trigger_pos_x = self.helper.config.get('trigger_pos_x', 2)  # 默认触发器位置x
        trigger_pos_y = self.helper.config.get('trigger_pos_y', 2)  # 默认触发器位置y

        self.mask[:, :, trigger_pos_x:trigger_pos_x + trigger_size, trigger_pos_y:trigger_pos_y + trigger_size] = 1
        self.trigger0 = self.trigger.clone()
        # 将触发器设置成从2，2开始，尺寸为5，红色的
        if self.helper.config.attack_method == 'Neurotoxin':
            # 设置触发器为红色：红通道=1（如果是归一化的值），其他通道=0
            red_value = 1.0  # 假设值已经归一化到 [0, 1]
            self.trigger[:, 0, trigger_pos_x:trigger_pos_x + trigger_size,
            trigger_pos_y:trigger_pos_y + trigger_size] = red_value
            self.trigger[:, 1, trigger_pos_x:trigger_pos_x + trigger_size,
            trigger_pos_y:trigger_pos_y + trigger_size] = 0.0
            self.trigger[:, 2, trigger_pos_x:trigger_pos_x + trigger_size,
            trigger_pos_y:trigger_pos_y + trigger_size] = 0.0
        if self.helper.config.attack_method == 'CerP':
            trigger_value = self.helper.config.get('trigger_value', 1)  # 默认触发器值为0.5
            self.trigger0 = torch.ones((1, channels, height, width), requires_grad=False, device=device) * trigger_value
            self.noise_trigger = self.trigger0

    def init_badnets_trigger(self):
        print('Setup baseline trigger pattern.')
        self.trigger[:, 0, :,:] = 1
        return

    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()

        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.helper.config.dm_adv_epochs):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                benign_inputs = inputs
                inputs = trigger * mask + (1 - mask) * inputs
                outputs = adv_model(inputs)
                if self.helper.config.attack_method == 'A3FL':
                    loss = ce_loss(outputs, labels)
                elif self.helper.config.attack_method == 'My':
                    benign_outputs = adv_model(benign_inputs)
                    loss = ce_loss(outputs, labels) + ce_loss(benign_outputs, labels)

                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum / sim_count

    def search_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        adv_models = []
        adv_ws = []

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t * m + (1 - m) * inputs
                    labels[:] = self.helper.config.target_class
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct / num_data
            return asr, total_loss

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config.trigger_lr

        K = self.helper.config.trigger_outter_epochs
        t = self.trigger.clone()
        m = self.mask.clone()

        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()

        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr=alpha * 10, weight_decay=0)
        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            if iter % self.helper.config.dm_adv_K == 0 and iter != 0:
                if len(adv_models) > 0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.helper.config.dm_adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t * m + (1 - m) * inputs
                labels[:] = self.helper.config.target_class
                outputs = model(inputs)
                loss = ce_loss(outputs, labels)

                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.helper.config.noise_loss_lambda * adv_w * nm_loss / self.helper.config.dm_adv_model_count
                        else:
                            loss += self.helper.config.noise_loss_lambda * adv_w * nm_loss / self.helper.config.dm_adv_model_count
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()

    def my_search_trigger(self, model, dl, type_, adversary_id=0, epoch=0):
        trigger_optim_time_start = time.time()
        K = 0
        model.eval()
        adv_models = []
        adv_ws = []

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t * m + (1 - m) * inputs
                    labels[:] = self.helper.config.target_class
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1]
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct / num_data
            return asr, total_loss

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config.trigger_lr

        K = self.helper.config.trigger_outter_epochs
        t = self.trigger.clone()
        m = self.mask.clone()

        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()

        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr=alpha * 10, weight_decay=0)
        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            if iter % self.helper.config.dm_adv_K == 0 and iter != 0:
                if len(adv_models) > 0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.helper.config.dm_adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t * m + (1 - m) * inputs
                labels[:] = self.helper.config.target_class
                outputs = model(inputs)
                loss = ce_loss(outputs, labels)

                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        model_params_diff = 0
                        for param_local, param_global in zip(adv_model.parameters(), model.parameters()):
                            model_params_diff += torch.norm(param_local - param_global)
                        loss += model_params_diff * 0.00001
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.helper.config.noise_loss_lambda * adv_w * nm_loss / self.helper.config.dm_adv_model_count
                        else:
                            loss += self.helper.config.noise_loss_lambda * nm_loss / self.helper.config.dm_adv_model_count
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()
    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])

        # 创建输入张量的副本，以避免原地操作
        poisoned_inputs = inputs.clone()

        # 应用触发器
        poisoned_inputs[:bkd_num] = self.trigger * self.mask + poisoned_inputs[:bkd_num] * (1 - self.mask)
        labels[:bkd_num] = self.helper.config.target_class

        return poisoned_inputs, labels


    def DBA_poison_input(self, inputs, labels, poison_pattern, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        # 确保 inputs 是至少三维的（例如：batch_size x height x width x channels）
        poisoned_inputs = inputs.clone()

        # 遍历 poison_pattern 中的坐标，并将对应位置的 RGB 通道值设为 1
        for i in range(bkd_num):
            for pos in poison_pattern:
                x, y = pos  # 获取坐标 (x, y)

                # 依次修改 RGB 通道的对应位置
                poisoned_inputs[i, 0, x, y] = 1  # R 通道
                poisoned_inputs[i, 1, x, y] = 0  # G 通道
                poisoned_inputs[i, 2, x, y] = 0  # B 通道

        # 修改标签为目标类
        labels[:bkd_num] = self.helper.config.target_class
        return poisoned_inputs, labels
    def cerp_poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])

        # 创建输入张量的副本，以避免原地操作
        poisoned_inputs = inputs.clone()

        # 应用触发器
        poisoned_inputs[:bkd_num] = self.noise_trigger * self.mask + poisoned_inputs[:bkd_num] * (1 - self.mask)
        labels[:bkd_num] = self.helper.config.target_class

        return poisoned_inputs, labels
