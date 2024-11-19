import os
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import (
    CIFAR10, CIFAR100, ImageFolder
)
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from PIL import Image
from typing import Tuple, Optional, Dict, List


class TinyImageNet(Dataset):

    def __init__(
        self,
        root: str,
        train: bool,
        transform: transforms = None,
        class_to_idx: Dict[str, int] = None
    ):
        
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.num_classes = 200

        self.data, self.targets, self.class_to_idx = self.__build_truncated_dataset__()
    
    def __build_truncated_dataset__(self):

        data, targets = [], []

        if self.train:

            tinyimagenet_dataobj = ImageFolder(root=self.root, transform=self.transform)
            for path, label in tinyimagenet_dataobj.samples:

                image = Image.open(path).convert('RGB')
                data.append(image)
                targets.append(label)
                self.class_to_idx = tinyimagenet_dataobj.class_to_idx            
        else:

            self.images_dir = os.path.join(self.root, 'images')
            
            filename_to_classid = {}
            with open(os.path.join(self.root, 'val_annotations.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    filename_to_classid[parts[0]] = parts[1]

            for filename, classid in filename_to_classid.items():
                image_path = os.path.join(self.images_dir, filename)
                image = Image.open(image_path).convert('RGB')
                label = self.class_to_idx[classid]
                data.append(image)
                targets.append(label)
        
        targets = list(targets)

        return data, targets, self.class_to_idx

    def __getitem__(self, index: int):

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.data)


def create_cached_data_loaders(
    args: argparse.ArgumentParser,
    data_indices_per_client: List[List[List[int]]],
    num_classes: int
) -> Tuple[Dict[int, DataLoader], DataLoader]:
    
    num_clients = args.num_clients
    dataset_type = args.dataset_type

    # CIFAR-10
    if dataset_type == 'cifar10':

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            )
        ])
        full_train_dataset = CIFAR10(
            root=args.datasets_dir, train=True, transform=transform, download=False
        )
        test_dataset = CIFAR10(
            root=args.datasets_dir, train=False, transform=transform, download=False
        )

    # CIFAR-100
    elif dataset_type == 'cifar100':

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]
            )
        ])
        full_train_dataset = CIFAR100(
            root=args.datasets_dir, train=True, transform=transform, download=False
        )
        test_dataset = CIFAR100(
            root=args.datasets_dir, train=False, transform=transform, download=False
        )
    
    #Tiny-ImageNet
    elif dataset_type == 'tinyimagenet':
        
        train_dir = os.path.join(args.datasets_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(args.datasets_dir, 'tiny-imagenet-200', 'val')

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )            
        ])
        full_train_dataset = TinyImageNet(
            root=train_dir, train=True, transform=transform, class_to_idx=None
        )
        test_dataset = TinyImageNet(
            root=val_dir, train=False, transform=transform, class_to_idx=full_train_dataset.class_to_idx
        )        
 
    train_loaders = {}
    all_indices = set()
    full_indices = set(range(len(full_train_dataset.targets)))
    for clt in range(num_clients):
        indices = get_full_indices(
            full_train_dataset.targets, data_indices_per_client[clt], num_classes
        )
        # 重複チェック
        if len(all_indices.intersection(indices)) > 0:
            print(f'重複発見: クライアント {clt} のデータと他のクライアントのデータが重複しています。')
        all_indices.update(indices)
        client_dataset = Subset(full_train_dataset, indices)
        train_loaders[clt] = DataLoader(
            dataset=client_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if all_indices != full_indices:
         print("エラー: クライアントに割り当てられていないインデックスがあります。")
    else:
        print("OK: 全てのクライアントに重複なくインデックスが割り当てられました")

    return train_loaders, test_loader


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
