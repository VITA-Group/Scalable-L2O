from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder
from torch.utils.data import DataLoader, Subset

import os
import numpy as np

def cifar10_dataloaders(batch_size=128, data_dir = 'datasets/cifar10'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)

    return train_loader, test_loader

def cifar10_subset_dataloaders(batch_size=128, data_dir = 'datasets/cifar10'):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(4500)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=128, data_dir = 'datasets/cifar100'):
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    #val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return train_loader, test_loader

def fashionmnist_dataloaders(batch_size=64, data_dir = 'datasets/fashionmnist'):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = Subset(FashionMNIST(data_dir, train=True, transform=train_transform, download=True), list(range(55000)))
    val_set = Subset(FashionMNIST(data_dir, train=True, transform=test_transform, download=True), list(range(55000, 60000)))
    test_set = FashionMNIST(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return train_loader, val_loader, test_loader


def tiny_imagenet_dataloaders(batch_size=64, data_dir = 'datasets/tiny-imagenet-200', dataset=False, split_file=None):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    if not split_file:
        split_file = 'npy_files/tiny-imagenet-train-val.npy'
    split_permutation = list(np.load(split_file))

    train_set = Subset(ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = Subset(ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = ImageFolder(val_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    if dataset:
        print('return train dataset')
        train_dataset = ImageFolder(train_path, transform=train_transform)
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader
