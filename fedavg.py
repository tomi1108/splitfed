import argparse
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import (
    mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152
)
from copy import deepcopy

from tqdm import tqdm
from typing import Tuple, Dict, List
from save_results import *
from datasetting import *
from datasets import *

# cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# bool型変数に変換
def str_to_bool(value):
    if isinstance(value, str):
        if value == 'True':
            return True
        elif value == 'False':
            return False
    else:
        raise ValueError(f"Cannot conver {value} to bool")

# seed
def set_seed(seed: int, device):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def fedavg(
    client_weights: Dict[int, Dict[str, torch.Tensor]],
    fedavg_ratios: Dict[int, float]
) -> nn.Module:

    global_weights = copy.deepcopy(client_weights[0])
    for k in global_weights.keys():
        global_weights[k] = global_weights[k] * fedavg_ratios[0]
        for client_id in range(1, len(client_weights)):
            global_weights[k] += client_weights[client_id][k] * fedavg_ratios[client_id]
    
    return global_weights

class client_resnet50(nn.Module):

    def __init__(self, model, num_classes, projected_size):
        super(client_resnet50, self).__init__()

        input_size = model.layer4[2].conv3.out_channels

        self.base_encoder = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.projection_head = nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            nn.Linear(input_size, projected_size)
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(projected_size),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(projected_size, num_classes, bias=True)
        )
    
    def forward(self, x):

        f_conv = self.base_encoder(x)
        f_proj = self.projection_head(f_conv)
        o = self.output_layer(f_proj)

        return f_conv, f_proj, o

def get_model_constructor(model_type: str):

    model_mapping = {
        'resnet50': (resnet50, client_resnet50)
    }

    return model_mapping.get(model_type)

def create_model(
    args: argparse.ArgumentParser,
    num_clients: int,
    num_classes: int,
    device: torch.device
):
    
    model_type = args.model_type
    dataset_type = args.dataset_type
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    
    if dataset_type == 'cifar10':
        projected_size = 128
    if dataset_type == 'cifar100':
        projected_size = 256
    if dataset_type == 'tinyimagenet':
        projected_size = 512

    model_constructor = get_model_constructor(model_type)
    base_mc, client_mc = model_constructor
    base_model = base_mc(weights=None).to(device)
    client_models = {
        client_id: client_mc(deepcopy(base_model), num_classes, projected_size)
        for client_id in range(num_clients)
    }
    client_optimizers = {
        client_id: optim.SGD(client_models[client_id].parameters(), lr, momentum, weight_decay) for client_id in range(num_clients)
    }
    client_schedulers = {
            client_id: CosineAnnealingLR(
            optimizer = client_optimizers[client_id], 
            T_max = args.num_rounds,
            eta_min = args.min_lr,
            last_epoch = -1
        ) for client_id in range(num_clients)
    }

    return client_models, client_optimizers, client_schedulers

def train_client(
    args: argparse.ArgumentParser,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
):
    
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(args.num_epochs):

        for images, labels in tqdm(train_loader):

            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            b_outs, p_outs, o_outs = model(images)
            loss = criterion(o_outs, labels)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()


def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 

# parser
parser = argparse.ArgumentParser(description='federated learning')
parser.add_argument('--app_name', type=str, default='FL', help='the approach name of this setting')
parser.add_argument('--seed', type=int, default=42, help='seed of numpy, random and torch')
parser.add_argument('--num_clients', type=int, default=2, help='the number of clients')
parser.add_argument('--num_rounds', type=int, default=50, help='the number of global rounds')
parser.add_argument('--num_epochs', type=int, default=5, help='the number of global epochs')
parser.add_argument('--projected_size', type=int, default=256, help='output size of projection head')
parser.add_argument('--batch_size', type=int , default=128, help='the batch_size of train dataloader')
parser.add_argument('--model_type', type=str, choices=['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='mobilenet_v2', help='the type of model using training')
parser.add_argument('--dataset_type', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet'], default='cifar10', help='the type of dataset using training')
parser.add_argument('--datasets_dir', type=str, default='~/datasets/', help='path to datasets directory')
parser.add_argument('--results_dir', type=str, default='./results/', help='path to results directory')
parser.add_argument('--data_dist_type', type=str, choices=['iid', 'non-iid'], default='iid', help='select the type of data distribution')
parser.add_argument('--alpha', type=float, default=1.0, help='parameter of dirichlet distribution')
parser.add_argument('--lr', type=float, default=0.1, help='the leraning rate of training')
parser.add_argument('--min_lr', type=float, default=0.00001, help='the minimum leraning rate of training')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum of training')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay of training')
parser.add_argument('--save_flag', type=str, default='False', help='whether to record the results')
args = parser.parse_args()
args.save_flag = str_to_bool(args.save_flag)


def main(args: argparse.ArgumentParser, device: torch.device):

    # save or not
    if args.save_flag:
        resutls_path = results_setting(args)
    else:
        print(f'args.save_flag: {args.save_flag}')

    num_clients = args.num_clients
    num_classes, data_size_per_client, data_indices_per_client = get_data_indices_per_client(args)
    fedavg_ratios = {client_id: data_size / sum(data_size_per_client) for client_id, data_size in enumerate(data_size_per_client)}
    print(fedavg_ratios)

    print('=== Data size ===')
    for clt in range(num_clients):
        print(f'Client {clt}: {data_size_per_client[clt]}')
    
    # create dataloader
    train_loaders, test_loader = create_cached_data_loaders(args, data_indices_per_client, num_classes)
    client_models, client_optimizers, client_schedulers = create_model(args, num_clients, num_classes, device)
    criterion = nn.CrossEntropyLoss()

    for round in range(args.num_rounds):

        start_time = time.time()
        print(f'=== Round[{round+1}/{args.num_rounds}] ===')
        for client_model in client_models.values():
            client_model.train()
        
        trained_client_models = {}
        for client_id in range(num_clients):
            trained_client_models[client_id] = train_client(args, device, client_models[client_id], client_optimizers[client_id], train_loaders[client_id])

        for client_model in client_models.values():
            client_model.eval()
        
        global_client_model = fedavg(trained_client_models, fedavg_ratios)
        for client_model in client_models.values():
            client_model.load_state_dict(global_client_model)

        for client_scheduler in client_schedulers.values():
            client_scheduler.step()

        correct = 0
        total = 0
        testing_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                _, _, outs = client_models[0](images)
                loss = criterion(outs, labels)
                testing_loss += loss.item()

                _, predicted = torch.max(outs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        testing_loss /= len(test_loader)
        print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {testing_loss:.4f}')

        end_time = time.time()
        print(f'this round takes {end_time-start_time} s.')

if __name__ == '__main__':

    set_seed(args.seed, device)
    print(args)
    main(args, device)