import os
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import (
    CIFAR10, CIFAR100, ImageFolder
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from typing import Tuple, Optional, Dict, List

class cifar10(Dataset):

    def __init__(
        self,
        datasets_dir: str, 
        data_indices: List[List[int]] = None,
        train: bool = True,
        transform: transforms = None,
        download = True
    ):
        
        self.root = datasets_dir
        self.data_indices = data_indices
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 10

        self.data, self.target = self.__build_truncated_dataset__()
    
    def __build_truncated_dataset__(self):

        cifar10_dataobj = CIFAR10(root=self.root, train=self.train, download=self.download)
        data = np.array(cifar10_dataobj.data)
        target = np.array(cifar10_dataobj.targets)

        if self.data_indices is not None:

            self.data_indices = get_full_indices(target, self.data_indices, self.num_classes)
            data = data[self.data_indices]
            target = target[self.data_indices]
        
        return data, target

    def __getitem__(self, index: int):

        img, target = self.data[index], self.target[index]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.data)
    

class cifar100(Dataset):

    def __init__(
        self,
        datasets_dir: str, 
        data_indices: List[List[int]] = None,
        train: bool = True,
        transform: transforms = None,
        download = True
    ):
        
        self.root = datasets_dir
        self.data_indices = data_indices
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 100

        self.data, self.target = self.__build_truncated_dataset__()
    
    def __build_truncated_dataset__(self):

        cifar10_dataobj = CIFAR100(root=self.root, train=self.train, download=self.download)
        data = np.array(cifar10_dataobj.data)
        target = np.array(cifar10_dataobj.targets)

        if self.data_indices is not None:

            self.data_indices = get_full_indices(target, self.data_indices, self.num_classes)
            data = data[self.data_indices]
            target = target[self.data_indices]
        
        return data, target

    def __getitem__(self, index: int):

        img, target = self.data[index], self.target[index]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.data)


class tinyimagenet(Dataset):

    def __init__(
        self,
        datasets_dir: str,
        data_indices: List[List[int]] = None,
        train: bool = True,
        transform: transforms = None,
        class_to_idx: Dict[str, int] = None
    ):
        
        self.root = os.path.expanduser(datasets_dir)
        self.data_indices = data_indices
        self.train = train
        self.transform = transform
        self.num_classes = 200
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            self.class_to_idx = None

        self.data, self.target, self.class_to_idx = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        if self.train:
            tinyimagenet_dataobj = ImageFolder(root=self.root, transform=self.transform)
            data, target = [], []
            for path, label in tqdm(tinyimagenet_dataobj.samples):
                image = Image.open(path).convert('RGB')
                data.append(image)
                target.append(label)
            self.class_to_idx = tinyimagenet_dataobj.class_to_idx

        else:
            images_dir = os.path.join(self.root, 'images')
            filename_to_classid = {}
            with open(os.path.join(self.root, 'val_annotations.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    filename = parts[0]
                    classid = parts[1]
                    filename_to_classid[filename] = classid

            data, target = [], []
            for filename, classid in filename_to_classid.items():
                image_path = os.path.join(images_dir, filename)
                image = Image.open(image_path).convert('RGB')
                label = self.class_to_idx[classid]
                data.append(image)
                target.append(label)
            
        target = np.array(target)

        if self.data_indices is not None:

            self.data_indices = get_full_indices(target, self.data_indices, self.num_classes)
            data = [data[i] for i in self.data_indices]
            target = target[self.data_indices]

        return data, target, self.class_to_idx
    
    def __getitem__(self, index:int):

        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    

def get_full_indices(
    targets: List[int],
    reduced_indices: List[List[int]],
    num_classes: int
) -> List[int]:
    
    full_indices = [ [] for _ in range(num_classes) ]
    for idx, target in enumerate(targets):
        full_indices[target].append(idx)

    use_indices = []
    for cls in range(num_classes):

        reduced_indices_of_cls = reduced_indices[cls]
        
        for idx in reduced_indices_of_cls:

            use_indices.append(full_indices[cls][idx])

    return use_indices


def create_data_loader(
    args: argparse.ArgumentParser,
    data_indices_per_client: List[List[List[int]]],
    num_classes: int
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    
    num_clients = args.num_clients
    dataset_type = args.dataset_type
    transform = transforms.ToTensor()
    train_datasets = {}
    train_loaders = {}

    if dataset_type == 'cifar10':

        for clt in range(num_clients):
            train_datasets[clt] = cifar10(
                datasets_dir=args.datasets_dir, 
                data_indices=data_indices_per_client[clt],
                train=True,
                transform=transform,
                download=False
            )
        test_dataset = cifar10(
            datasets_dir=args.datasets_dir,
            train=False,
            transform=transform,
            download=False
        )
    
    elif dataset_type == 'cifar100':

        for clt in range(num_clients):
            train_datasets[clt] = cifar100(
                datasets_dir=args.datasets_dir,
                data_indices=data_indices_per_client[clt],
                train=True,
                transform=transform,
                download=False
            )
        test_dataset = cifar100(
            datasets_dir=args.datasets_dir,
            train=False,
            transform=transform,
            download=False
        )
    
    elif dataset_type == 'tinyimagenet':

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )            
        ])
        train_dir = os.path.join(args.datasets_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(args.datasets_dir, 'tiny-imagenet-200', 'val')

        for clt in range(num_clients):
            train_datasets[clt] = tinyimagenet(
                datasets_dir=train_dir,
                data_indices=data_indices_per_client[clt],
                train=True,
                transform=transform
            )
        test_dataset = tinyimagenet(
            datasets_dir=val_dir,
            train=False,
            transform=transform,
            class_to_idx=train_datasets[0].class_to_idx
        )
        
    
    # for client_id, train_dataset in train_datasets.items():
    #     counts = {i: 0 for i in range(num_classes)}
    #     for target in train_dataset.target:
    #         counts[target] += 1
    #     print(f'Client {client_id}: {counts}')

    for clt in range(num_clients):
        train_loaders[clt] = DataLoader(dataset=train_datasets[clt], batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loaders, test_loader
