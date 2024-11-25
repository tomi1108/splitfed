import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision.models import (
    mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152
)
from copy import deepcopy

def get_model_constructor(model_type: str):

    model_mapping = {
        'mobilenet_v2': (mobilenet_v2, client_mobilenet_v2, server_mobilenet_v2),
        'resnet18': (resnet18, client_resnet18, server_resnet18),
        'resnet34': (resnet34, client_resnet34, server_resnet34),
        'resnet50': (resnet50, client_resnet50, server_resnet50),
        'resnet101': (resnet101, client_resnet101, server_resnet101),
        'resnet152': (resnet152, client_resnet152, server_resnet152)
    }

    return model_mapping.get(model_type)

def create_model(
    args: argparse.ArgumentParser,
    num_clients: int,
    num_classes: int,
    device: torch.device
):
    
    model_type = args.model_type
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    
    model_constructor = get_model_constructor(model_type)
    if model_constructor is None:
        raise ValueError(f'unsupported model type: {model_type}')
    
    base_mc, client_mc, server_mc = model_constructor
    base_model = base_mc(weights=None).to(device)
    
    client_models = {
        client_id: client_mc(deepcopy(base_model)) for client_id in range(num_clients)
    }
    server_model = server_mc(args, num_classes, deepcopy(base_model))

    client_optimizers = {
        client_id: optim.SGD(client_models[client_id].parameters(), lr, momentum, weight_decay) for client_id in range(num_clients)
    }
    client_schedulers = {
        # client_id: StepLR(client_optimizers[client_id], step_size=20, gamma=0.1, last_epoch=-1) for client_id in range(num_clients)
        client_id: CosineAnnealingLR(
            optimizer = client_optimizers[client_id], 
            T_max = args.num_rounds - args.warmup_rounds,
            eta_min = args.min_lr,
            last_epoch = -1
        ) for client_id in range(num_clients)
    }
    server_optimizer = optim.SGD(server_model.parameters(), lr, momentum, weight_decay)
    server_scheduler = CosineAnnealingLR(
        optimizer=server_optimizer,
        T_max = args.num_rounds - args.warmup_rounds,
        eta_min = args.min_lr,
        last_epoch = -1
    )
    # server_scheduler = StepLR(server_optimizer, step_size=20, gamma=0.1, last_epoch=-1)

    
    return client_models, server_model, client_optimizers, server_optimizer, client_schedulers, server_scheduler


# MobileNet系
class client_mobilenet_v2(nn.Module):

    def __init__(self, model):
        super(client_mobilenet_v2, self).__init__()
        
        self.model = model.features[:6]

    def forward(self, x):
        return self.model(x)

class server_mobilenet_v2(nn.Module):

    def __init__(self, args, num_classes, model):
        super(server_mobilenet_v2, self).__init__()

        input_size = model.features[-1][0].out_channels
        projected_size = args.projected_size

        self.base_encoder = model.features[6:]
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_size, projected_size)
        )
        self.output_layer = nn.Sequential(
            nn.ReLU6(inplace=False),
            nn.Linear(projected_size, num_classes, bias=True)
        )
    
    def forward(self, x):

        f_conv = self.base_encoder(x)
        f_proj = self.projection_head(f_conv)
        o = self.output_layer(f_proj)

        return f_conv, f_proj, o

'''
ResNet系のモデル定義 [resnet18, resnet34, resnet50, resnet101, resnet152]
'''
##### RESNET18 #####
class client_resnet18(nn.Module):

    def __init__(self, model):
        super(client_resnet18, self).__init__()
        
        self.model = nn.Sequential(
            model.conv1,
            model.relu,
            model.maxpool,
            model.layer1
        )

    def forward(self, x):
        return self.model(x)


class server_resnet18(nn.Module):

    def __init__(self, args, num_classes, model):
        super(server_resnet18, self).__init__()

        input_size = model.layer4[1].conv2.out_channels
        projected_size_dict = {
            'cifar10': 128,
            'cifar100': 256,
            'tinyimagenet': 512
        }
        projected_size = projected_size_dict[args.dataset_type]

        self.base_encoder = nn.Sequential(
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
            # nn.ReLU(inplace=False),
            # nn.Dropout(p=0.2, inplace=False),
            nn.Linear(projected_size, num_classes, bias=True)
        )
    
    def forward(self, x):

        f_conv = self.base_encoder(x)
        f_proj = self.projection_head(f_conv)
        o = self.output_layer(f_proj)

        return f_conv, f_proj, o


##### RESNET34 #####
class client_resnet34(nn.Module):

    def __init__(self, model):
        super(client_resnet34, self).__init__()
        
        self.model = nn.Sequential(
            model.conv1,
            model.relu,
            model.maxpool,
            model.layer1
        )

    def forward(self, x):
        return self.model(x)

class server_resnet34(nn.Module):

    def __init__(self, args, num_classes, model):
        super(server_resnet34, self).__init__()

        input_size = model.layer4[1].conv2.out_channels
        projected_size_dict = {
            'cifar10': 128,
            'cifar100': 256,
            'tinyimagenet': 512
        }
        projected_size = projected_size_dict[args.dataset_type]

        self.base_encoder = nn.Sequential(
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
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(projected_size, num_classes, bias=True)
        )
    
    def forward(self, x):

        f_conv = self.base_encoder(x)
        f_proj = self.projection_head(f_conv)
        o = self.output_layer(f_proj)

        return f_conv, f_proj, o

##### RESNET50 #####
class client_resnet50(nn.Module):

    def __init__(self, model):
        super(client_resnet50, self).__init__()
        
        self.model = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1
            # model.layer2
        )

    def forward(self, x):
        return self.model(x)

class server_resnet50(nn.Module):

    def __init__(self, args, num_classes, model):
        super(server_resnet50, self).__init__()

        input_size = model.layer4[2].conv3.out_channels
        projected_size = args.projected_size

        self.base_encoder = nn.Sequential(
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
            # nn.ReLU(inplace=False),
            # nn.Dropout(p=0.2, inplace=False),
            nn.BatchNorm1d(projected_size),
            nn.Linear(projected_size, num_classes, bias=True)
        )
    
    def forward(self, x):

        f_conv = self.base_encoder(x)
        f_proj = self.projection_head(f_conv)
        o = self.output_layer(f_proj)

        return f_conv, f_proj, o


##### RESNET101 #####
class client_resnet101(nn.Module):

    def __init__(self, model):
        super(client_resnet101, self).__init__()
        
        self.model = nn.Sequential(
            model.conv1,
            model.relu,
            model.maxpool,
            model.layer1
        )

    def forward(self, x):
        return self.model(x)

class server_resnet101(nn.Module):

    def __init__(self, args, num_classes, model):
        super(server_resnet101, self).__init__()

        input_size = model.layer4[2].conv3.out_channels
        projected_size_dict = {
            'cifar10': 128,
            'cifar100': 256,
            'tinyimagenet': 512
        }
        projected_size = projected_size_dict[args.dataset_type]

        self.base_encoder = nn.Sequential(
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.projection_head = nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            nn.Linear(input_size, projected_size * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(projected_size * 2, projected_size)
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(projected_size, num_classes, bias=True)
        )
    
    def forward(self, x):

        f_conv = self.base_encoder(x)
        f_proj = self.projection_head(f_conv)
        o = self.output_layer(f_proj)

        return f_conv, f_proj, o


##### RESNET152 #####
class client_resnet152(nn.Module):

    def __init__(self, model):
        super(client_resnet152, self).__init__()
        
        self.model = nn.Sequential(
            model.conv1,
            model.relu,
            model.maxpool,
            model.layer1
        )

    def forward(self, x):
        return self.model(x)

class server_resnet152(nn.Module):

    def __init__(self, args, num_classes, model):
        super(server_resnet152, self).__init__()

        input_size = model.layer4[2].conv3.out_channels
        projected_size_dict = {
            'cifar10': 128,
            'cifar100': 256,
            'tinyimagenet': 512
        }
        projected_size = projected_size_dict[args.dataset_type]

        self.base_encoder = nn.Sequential(
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.projection_head = nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            nn.Linear(input_size, projected_size * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(projected_size * 2, projected_size)
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(projected_size, num_classes, bias=True)
        )
    
    def forward(self, x):

        f_conv = self.base_encoder(x)
        f_proj = self.projection_head(f_conv)
        o = self.output_layer(f_proj)

        return f_conv, f_proj, o