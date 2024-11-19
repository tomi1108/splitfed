import argparse
import time
import random
import torch
import numpy as np
import copy
from tqdm import tqdm
from typing import Tuple, Dict, List
from datasets import *
from datasetting import *
from model import *
from save_results import *
from prototype import *

# cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

# parser
parser = argparse.ArgumentParser(description='splitfed learning')
parser.add_argument('--app_name', type=str, default='SFL', help='the approach name of this setting')
parser.add_argument('--seed', type=int, default=42, help='seed of numpy, random and torch')
parser.add_argument('--num_clients', type=int, default=2, help='the number of clients')
parser.add_argument('--num_rounds', type=int, default=50, help='the number of global rounds')
parser.add_argument('--num_epochs', type=int, default=5, help='the number of global epochs')
parser.add_argument('--batch_size', type=int , default=128, help='the batch_size of train dataloader')
parser.add_argument('--model_type', type=str, choices=['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], default='mobilenet_v2', help='the type of model using training')
parser.add_argument('--dataset_type', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet'], default='cifar10', help='the type of dataset using training')
parser.add_argument('--datasets_dir', type=str, default='~/datasets/', help='path to datasets directory')
parser.add_argument('--results_dir', type=str, default='./results/', help='path to results directory')
parser.add_argument('--data_dist_type', type=str, choices=['iid', 'non-iid'], default='iid', help='select the type of data distribution')
parser.add_argument('--alpha', type=float, default=1.0, help='parameter of dirichlet distribution')
parser.add_argument('--threshold', type=float, default=0.1, help='threshold of positive or negative in P_SFL')
parser.add_argument('--u', type=float, default=0.5, help='hyperparameter about prototype effective')
parser.add_argument('--lr', type=float, default=0.01, help='the leraning rate of training')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum of training')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay of training')
parser.add_argument('--save_flag', type=str, default='False', help='whether to record the results')
args = parser.parse_args()
args.save_flag = str_to_bool(args.save_flag)

set_seed(args.seed, device)


class Server:
    def __init__(
            self,
            args: argparse.ArgumentParser,
            model: nn.Module,
            test_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            device: torch.device
        ):

        self.args = args
        self.device = device
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.optimizer = optimizer
    
        self.criterion = nn.CrossEntropyLoss()
        self.running_loss = 0.0

    def train(
        self,
        intermediates: Tuple[int, torch.Tensor, torch.Tensor],
        prototypes: Prototypes = None,
    ) -> Dict[int, torch.Tensor]:
        
        client_id = intermediates[0]
        smashed_data = intermediates[1].clone().detach()
        labels = intermediates[2]

        smashed_data.requires_grad_(True)
        smashed_data.retain_grad()

        self.optimizer.zero_grad()
        f_conv, f_proj, outs = self.model(smashed_data)
        loss = self.criterion(outs, labels)
        self.running_loss += loss.item()

        if self.args.app_name == 'P_SFL':
            if prototypes.flag:
                p_loss = prototypes.calculate_loss(client_id, smashed_data, labels)
                loss = loss + args.u * p_loss
        # if self.args.app_name == 'PM_SFL':
        #     if prototypes.flag:
        #         p_loss = prototypes.calculate_loss_pm(client_id, f_proj, labels)
        #         loss = loss + args.u * p_loss

        loss.backward()
        self.optimizer.step()

        if self.args.app_name == 'P_SFL':
            if prototypes.flag: prototypes.sub_optimizers[client_id].step()

        gradients = {client_id: smashed_data.grad}
    
        return gradients

    
    def evaluate(
        self,
        client_model: nn.Module
    ):
        
        correct = 0
        total = 0
        testing_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                _, _, outs = self.model(client_model(images))
                loss = self.criterion(outs, labels)
                testing_loss += loss.item()

                _, predicted = torch.max(outs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        testing_loss /= len(self.test_loader)
        print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {testing_loss:.4f}')
        return accuracy, testing_loss


class Client:
    def __init__(
            self,
            client_id: int,
            model: nn.Module,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            device: torch.device
        ):

        self.client_id = client_id
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.data_iterator = iter(self.train_loader)
        self.optimizer = optimizer

        self.criterion = nn.CrossEntropyLoss()

    def forward(self):

        try:
            images, labels = next(self.data_iterator)
        except:
            self.data_iterator = iter(self.train_loader)
            images, labels = next(self.data_iterator)
        
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        self.smashed_data = self.model(images)

        return self.client_id, self.smashed_data, labels

    def backward(
        self,
        grad: Dict[int, torch.Tensor]
    ):

        self.optimizer.zero_grad()
        self.smashed_data.grad = grad[self.client_id].clone().detach()
        self.smashed_data.backward(gradient=self.smashed_data.grad)
        self.optimizer.step()
    
    def evaluate(
        self,
        server_model,
        prototypes: Prototypes = None
    ):

        train_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                f_conv, f_proj, outs = server_model(self.model(images))
                loss = self.criterion(outs, labels)
                train_loss += loss.item()

                _, predicted = torch.max(outs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                if args.app_name == 'P_SFL':
                    prototypes.save_projected(f_proj, outs, labels)
                # if args.app_name == 'PM_SFL':
                #     prototypes.save_projected(f_proj, outs, labels)
                    
            train_loss /= len(self.train_loader)
        
        return correct, total, train_loss


def fedavg(
    client_weights: Dict[int, Dict[str, torch.Tensor]],
    fedavg_ratios: Dict[int, float]
) -> nn.Module:

    global_weights = copy.deepcopy(client_weights[0])
    for k in global_weights.keys():
        global_weights[k] = global_weights[k] * fedavg_ratios[0]
        for client_id in range(1, len(client_weights)):
            global_weights[k] += client_weights[client_id][k] * fedavg_ratios[client_id]
        # global_weights[k] = torch.div(global_weights[k], len(client_weights))
    
    return global_weights


def main(args: argparse.ArgumentParser, device: torch.device):

    # seed
    # set_seed(args.seed, device)

    # save or not
    if args.save_flag:
        resutls_path = results_setting(args)
    else:
        print(f'args.save_flag: {args.save_flag}')

    # data indices setting
    num_clients = args.num_clients
    num_classes, data_size_per_client, data_indices_per_client = get_data_indices_per_client(args)
    fedavg_ratios = {client_id: data_size / sum(data_size_per_client) for client_id, data_size in enumerate(data_size_per_client)}
    print(fedavg_ratios)

    print('=== Data size ===')
    for clt in range(num_clients):
        print(f'Client {clt}: {data_size_per_client[clt]}')
    num_iter = int((sum(data_size_per_client) / args.batch_size) // num_clients)
    print(f'The number of iterations per global epoch: {num_iter}')

    # create dataloader
    train_loaders, test_loader = create_data_loader(args, data_indices_per_client, num_classes)
    client_models, server_model, client_optimizers, server_optimizer = create_model(args, num_clients, num_classes, device)        

    # server and client
    server = Server(args, server_model, test_loader, server_optimizer, device)
    clients = {
        client_id: Client(client_id, client_models[client_id], train_loaders[client_id], client_optimizers[client_id], device)
        for client_id in range(num_clients)
    }

    # prototype
    if args.app_name == 'P_SFL':
        prototypes = Prototypes(args, num_classes, device)
    # if args.app_name == 'PM_SFL':
    #     prototypes = Prototypes(args, num_classes, device)

    # training    
    for round in range(args.num_rounds):

        print(f'=== Round[{round+1}/{args.num_rounds}] ===')

        server.model.train()
        for client_id, client in clients.items():
            client.model.train()

        for epoch in range(args.num_epochs):

            # print(f'=== Epoch[{epoch+1}/{args.num_epochs}] ===')
            server.running_loss = 0.0

            for iter in tqdm(range(num_iter)):

                for client_id, client in clients.items():
                    intermediates = clients[client_id].forward()
                    if args.app_name == 'SFL': gradients = server.train(intermediates)
                    elif args.app_name == 'P_SFL': gradients = server.train(intermediates, prototypes)
                    # elif args.app_name == 'PM_SFL': gradients = server.train(intermediates, prototypes)
                    client.backward(gradients)
            
            server.running_loss = server.running_loss / (num_iter * num_clients)
            prRed(f'[Round {round+1}/Epoch {epoch+1}] Training Loss: {server.running_loss:.4f}')
        
        # test mode
        server.model.eval()
        for client_id, client in clients.items():
            client.model.eval()

        # fedavg
        trained_client_models = {}
        for client_id, client in clients.items():
            trained_client_models[client_id] = client.model.state_dict()
        global_client_model = fedavg(trained_client_models, fedavg_ratios)
        for client_id, client in clients.items():
            client.model.load_state_dict(global_client_model)
        
        # test evaluate
        test_accuracy, test_loss = server.evaluate(clients[0].model)

        # train evaluate
        corrects = 0
        totals = 0
        train_loss = 0.0
        for client_id, client in clients.items():
            if args.app_name == 'SFL': correct, total, loss = client.evaluate(server.model)
            elif args.app_name == 'P_SFL': correct, total, loss = client.evaluate(server.model, prototypes)
            # elif args.app_name == 'PM_SFL': correct, total, loss = client.evaluate(server.model, prototypes)
            corrects += correct
            totals += total
            train_loss += loss
        train_accuracy = corrects / totals * 100
        train_loss /= num_clients
        print(f'Train Accuracy: {train_accuracy:.2f}, Train Loss: {train_loss:.4f}')
        
        # save results
        if args.save_flag:
            header = [round+1, train_loss, test_loss, train_accuracy, test_accuracy]
            save_data(resutls_path, header)
        
        # calculate prototypes
        if args.app_name == 'P_SFL':
            prototypes.calculate_prototypes()
            prototypes.reset()
            # fedavg of sub projection head
            trained_sub_projection_heads = {}
            sub_fedavg_ratios = { client_id: 1 / num_clients for client_id in range(num_clients) }
            for client_id in range(num_clients):
                trained_sub_projection_heads[client_id] = prototypes.sub_projection_heads[client_id].state_dict()
            global_sub_projection_head = fedavg(trained_sub_projection_heads, sub_fedavg_ratios)
            for client_id in range(num_clients):
                prototypes.sub_projection_heads[client_id].load_state_dict(global_sub_projection_head)

        # if args.app_name == 'PM_SFL':
        #     prototypes.calculate_prototypes()
        #     prototypes.reset()


if __name__ == '__main__':

    print(args)
    main(args, device)
