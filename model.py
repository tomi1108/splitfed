import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    mobilenet_v2, resnet18, resnet34, resnet50, resnet101, resnet152
)
from copy import deepcopy

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
    
    if model_type == 'mobilenet_v2':

        base_mobilenet_v2 = mobilenet_v2(weights=None).to(device)
        client_models = {client_id: client_mobilenet_v2(deepcopy(base_mobilenet_v2)) for client_id in range(num_clients)}
        server_model = server_mobilenet_v2(args, num_classes, deepcopy(base_mobilenet_v2))

    elif model_type == 'resnet18':

        base_resnet18 = resnet18(weights=None).to(device)
        client_models = {client_id: client_resnet18(deepcopy(base_resnet18)) for client_id in range(num_clients)}
        server_model = server_resnet18(args, num_classes, deepcopy(base_resnet18))

    elif model_type == 'resnet34':

        base_resnet34 = resnet34(weights=None).to(device)
        client_models = {client_id: client_resnet34(deepcopy(base_resnet34)) for client_id in range(num_clients)}
        server_model = server_resnet34(args, num_classes, deepcopy(base_resnet34))

    elif model_type == 'resnet50':

        base_resnet50 = resnet50(weights=None).to(device)
        client_models = {client_id: client_resnet50(deepcopy(base_resnet50)) for client_id in range(num_clients)}
        server_model = server_resnet50(args, num_classes, deepcopy(base_resnet50))

    elif model_type == 'resnet101':

        base_resnet101 = resnet101(weights=None).to(device)
        client_models = {client_id: client_resnet101(deepcopy(base_resnet101)) for client_id in range(num_clients)}
        server_model = server_resnet101(args, num_classes, deepcopy(base_resnet101))

    elif model_type == 'resnet152':

        base_resnet152 = resnet152(weights=None).to(device)
        client_models = {client_id: client_resnet152(deepcopy(base_resnet152)) for client_id in range(num_clients)}
        server_model = server_resnet152(args, num_classes, deepcopy(base_resnet152))

    
    client_optimizers = {
        client_id: optim.SGD(params=client_models[client_id].parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        for client_id in range(num_clients)
    }
    server_optimizer = optim.SGD(params=server_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    return client_models, server_model, client_optimizers, server_optimizer


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
        projected_size_dict = {
            'cifar10': 128,
            'cifar100': 256,
            'tinyimagenet': 512
        }
        projected_size = projected_size_dict[args.dataset_type]

        self.base_encoder = model.features[6:]
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_size, projected_size)
        )
        self.output_layer = nn.Sequential(
            nn.ReLU6(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
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
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
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
            model.relu,
            model.maxpool,
            model.layer1
        )

    def forward(self, x):
        return self.model(x)

class server_resnet50(nn.Module):

    def __init__(self, args, num_classes, model):
        super(server_resnet50, self).__init__()

        input_size = model.layer4[2].conv2.out_channels
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