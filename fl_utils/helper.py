import sys

sys.path.append("../")
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from PIL import Image

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18, ResNet34, ResNet50
from torch.utils.data import Subset

class Helper:
    def __init__(self, config):
        self.config = config

        self.config.data_folder = './datasets'
        self.local_model = None
        self.global_model = None
        self.client_models = []
        self.setup_all()
        
        self.poison_pattern_0 = [[2, 2], [2, 3], [3, 2]]
        self.poison_pattern_1 = [[3, 3], [2, 4], [2, 5]]
        self.poison_pattern_2 = [[2, 6], [3, 4], [3, 5]]
        self.poison_pattern_3 = [[3, 6], [4, 2], [4, 3]]
        self.poison_pattern_4 = [[5, 2], [5, 3], [6, 2]]
        self.poison_pattern_5 = [[4, 4], [4, 5]]
        self.poison_pattern_6 = [[4, 6], [5, 4]]
        self.poison_pattern_7 = [[5, 5], [5, 6]]
        self.poison_pattern_8 = [[6, 3], [6, 4]]
        self.poison_pattern_9 = [[6, 5], [6, 6]]
        
        # self.poison_pattern_0 = [[2, 2], [2, 3], [3, 2], [3, 3]]
        # self.poison_pattern_1 = [[2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6]]
        # self.poison_pattern_2 = [[4, 2], [4, 3], [5, 2], [5, 3], [6, 2]]
        # self.poison_pattern_3 = [[4, 4], [4, 5], [4, 6], [5, 4], [5, 5], [5, 6], [6, 3], [6, 4], [6, 5], [6, 6]]

        # self.poison_pattern_0 = [[2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [4, 2],
        #                          [4, 3], [4, 4], [4, 5], [4, 6]]
        # self.poison_pattern_1 = [[5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6]]

        # self.poison_pattern_0 = [[2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [6, 2], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [6, 3], [4, 2], [4, 3],[4, 4], [4, 5], [4, 6], [6, 4], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [6, 5], [6, 6]]

    def setup_all(self):
        # 加载数据
        self.load_data()
        # 加载模型
        self.load_model()
        # 敌手列表
        self.config_adversaries()

    def load_model(self):
        # 动态选择输入通道数和类别数
        input_channels = 1 if self.config.dataset == 'MNIST' else 3
        num_classes = self.num_classes

        # 根据配置选择模型架构
        def get_model():
            if self.config.model == 'ResNet18':
                return ResNet18(num_classes=num_classes, in_channels=input_channels)
            elif self.config.model == 'ResNet34':
                return ResNet34(num_classes=num_classes, in_channels=input_channels)
            elif self.config.model == 'ResNet50':
                return ResNet50(num_classes=num_classes, in_channels=input_channels)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model}")

        # 选择设备：如果有CUDA可用，使用GPU；否则使用CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载本地模型
        self.local_model = get_model().to(device)

        # 加载全局模型
        self.global_model = get_model().to(device)

        # 加载参与者模型
        for _ in range(self.config.num_total_participants):
            t_model = get_model().to(device)
            self.client_models.append(t_model)

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config.num_worker)
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)

        return test_loader

    def load_data(self):
        if self.config.dataset == 'cifar10':
            self.num_classes = 10
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.train_dataset = datasets.CIFAR10(
                self.config.data_folder, train=True,
                download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(
                self.config.data_folder, train=False, transform=transform_test)

            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)

            train_loaders = [self.get_train(indices)
                             for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
            repair_size = getattr(self.config, "N_num", 2048)
            self.repair_data = self.build_repair_loader(repair_size=repair_size)

        elif self.config.dataset == 'cifar100':
            self.num_classes = 100
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
            ])
            self.train_dataset = datasets.CIFAR100(
                self.config.data_folder, train=True,
                download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR100(
                self.config.data_folder, train=False, transform=transform_test)

            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)

            train_loaders = [self.get_train(indices)
                             for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
            repair_size = getattr(self.config, "N_num", 2048)
            self.repair_data = self.build_repair_loader(repair_size=repair_size)

        elif self.config.dataset == 'FEMNIST':
            # Assuming the existence of a FEMNIST dataset implementation
            self.num_classes = 62  # 10 digits + 26 lowercase + 26 uppercase letters
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            # Replace with the actual dataset loading logic
            self.train_dataset = FEMNISTDataset(
                self.config.data_folder, train=True,
                download=True, transform=transform)
            self.test_dataset = FEMNISTDataset(
                self.config.data_folder, train=False, transform=transform)

            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)

            train_loaders = [self.get_train(indices)
                             for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
            repair_size = getattr(self.config, "N_num", 2048)
            self.repair_data = self.build_repair_loader(repair_size=repair_size)

        elif self.config.dataset == 'TinyImageNet':
            # Assuming the existence of a TinyImageNet dataset implementation
            self.num_classes = 200
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ])
            # Replace with the actual dataset loading logic
            self.train_dataset = TinyImageNet(
                self.config.data_folder, train=True,
                download=True, transform=transform_train)
            self.test_dataset = TinyImageNet(
                self.config.data_folder, train=False, transform=transform_test)

            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)

            train_loaders = [self.get_train(indices)
                             for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
            repair_size = getattr(self.config, "N_num", 2048)
            self.repair_data = self.build_repair_loader(repair_size=repair_size)

        elif self.config.dataset == 'MNIST':
            self.num_classes = 10
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            self.train_dataset = datasets.MNIST(
                self.config.data_folder, train=True,
                download=True, transform=transform)
            self.test_dataset = datasets.MNIST(
                self.config.data_folder, train=False, transform=transform)

            indices_per_participant = self.sample_dirichlet_train_data(
                self.config.num_total_participants,
                alpha=self.config.dirichlet_alpha)

            train_loaders = [self.get_train(indices)
                             for pos, indices in indices_per_participant.items()]

            self.train_data = train_loaders
            self.test_data = self.get_test()
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_worker)
            repair_size = getattr(self.config, "N_num", 2048)
            self.repair_data = self.build_repair_loader(repair_size=repair_size)


    def config_adversaries(self):
        if self.config.is_poison:
            self.adversary_list = list(range(self.config.num_adversaries))
        else:
            self.adversary_list = list()
    
    def build_repair_loader(self, repair_size=2048):
        # 从训练集中随机抽一个小子集作为“检测/修复”用的干净数据
        all_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(all_idx)
        idx = all_idx[:repair_size]
        subset = Subset(self.train_dataset, idx)
        return DataLoader(
            subset,
            batch_size=self.config.test_batch_size,   # 或者单独设 repair_batch_size
            shuffle=False,
            num_workers=self.config.num_worker
        )