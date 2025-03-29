import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_intermediate_info(args):

    intermediate_info_dict = {
        'mobilenet_v2': 
            {
                'cifar10': (32, 8, 8),
                'cifar100': (32, 8, 8),
                'tinyimagenet': (32, 8, 8)
            },
        'resnet18':
            {
                'cifar10': (64, 8, 8),
                'cifar100': (64, 8, 8),
                'tinyimagenet': (64, 8, 8)                
            },
        'resnet50':
            {
                'cifar10': (256, 16, 16),
                'cifar100': (256, 16, 16),
                'tinyimagenet': (256, 16, 16),
            }
    }

    return intermediate_info_dict[args.model_type][args.dataset_type]

class client_classifier(nn.Module):

    def __init__(self, smashed_data_size, projected_size):
        super(client_classifier, self).__init__()

        c, h, w = smashed_data_size
        input_size = c * h * w

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, projected_size)
    
    def forward(self, x):

        return self.fc(self.flatten(x))


class moon_trainner:

    def __init__(
        self,
        args: argparse.ArgumentParser,
        device: torch.device
    ):
        
        self.args = args
        self.device = device
        self.mu = self.args.mu
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        self.temperature = 0.5

        self.projected_size = self.args.projected_size
        self.smashed_data_size = get_intermediate_info(args)
        
        # 学習する全結合層を定義
        self.train_classifiers = {
            client_id: client_classifier(self.smashed_data_size, self.projected_size).to(self.device)
            for client_id in range(self.args.num_clients)
        }
        self.train_optimisers = {
            client_id: torch.optim.SGD(
                params = self.train_classifiers[client_id].parameters(),
                lr = self.args.lr,
                momentum = self.args.momentum,
                weight_decay = self.args.weight_decay
            )
            for client_id in range(self.args.num_clients)
        }
        self.train_schedulers = {
            client_id: CosineAnnealingLR(
                optimizer = self.train_optimisers[client_id],
                T_max = self.args.num_rounds - self.args.warmup_rounds,
                eta_min = self.args.min_lr,
                last_epoch = -1
            )
            for client_id in range(self.args.num_clients)
        }

        self.glob_classifiers = {
            client_id: client_classifier(self.smashed_data_size, self.projected_size).to(self.device)
            for client_id in range(self.args.num_clients)
        }
        self.prev_classifiers = {
            client_id: client_classifier(self.smashed_data_size, self.projected_size).to(self.device)
            for client_id in range(self.args.num_clients)
        }
    
    def forward(self, intermediates):

        client_id, smashed_data, _ = intermediates

        self.train_optimisers[client_id].zero_grad()
        projected = self.train_classifiers[client_id](smashed_data)
        glob_projected = self.glob_classifiers[client_id](smashed_data)
        prev_projected = self.prev_classifiers[client_id](smashed_data)
        # projected = F.normalize(self.train_classifiers[client_id](smashed_data), dim=1)
        # glob_projected = F.normalize(self.glob_classifiers[client_id](smashed_data), dim=1)
        # prev_projected = F.normalize(self.prev_classifiers[client_id](smashed_data), dim=1)

        pos = self.cosine(projected, glob_projected)
        neg = self.cosine(projected, prev_projected)

        logits = torch.cat((pos.reshape(-1, 1), neg.reshape(-1, 1)), dim=1)
        logits /= self.temperature
        logits_labels = torch.zeros(self.args.batch_size).to(self.device).long()

        loss = self.mu * self.criterion(logits, logits_labels)

        return loss
    
    def update(self, fedavg_ratios):

        global_weights = copy.deepcopy(self.train_classifiers[0].state_dict())
        for k in global_weights.keys():
            global_weights[k] = global_weights[k] * fedavg_ratios[0]
            for client_id in range(1, self.args.num_clients):
                global_weights[k] += self.train_classifiers[client_id].state_dict()[k] * fedavg_ratios[client_id]
        
        for glob_classifier in self.glob_classifiers.values():
            glob_classifier.load_state_dict(global_weights)
            for param in glob_classifier.parameters():
                param.requires_grad = False

        self.prev_classifiers = copy.deepcopy(self.train_classifiers)
        for prev_classifier in self.prev_classifiers.values():
            for param in prev_classifier.parameters():
                param.requires_grad = False

        for train_classifier in self.train_classifiers.values():
            train_classifier.load_state_dict(global_weights)

        for train_scheduler in self.train_schedulers.values():
            train_scheduler.step()