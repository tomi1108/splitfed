import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prototypes:

    def __init__(
        self,
        args: argparse.ArgumentParser,
        num_classes: int,
        device: torch.device
    ):
        
        self.args = args
        self.num_classes = num_classes
        self.device = device

        self.threshold = args.threshold
        self.batch_size = args.batch_size
        self.dataset_type = args.dataset_type

        # クラスごとの閾値のリスト
        self.class_threshold = [ self.threshold for _ in range(self.num_classes) ]
        
        self.positive_features = []
        self.positive_labels = []
        self.negative_features = []
        self.negative_labels = []

        self.data_counts = 0
        self.positive_counts = 0
        self.negative_counts = 0
        self.flag = False
        self.start = False
        self.pos_start = False
        self.neg_start = False

        projected_size_dict = {
            'cifar10': 128,
            'cifar100': 256,
            'tinyimagenet': 512
        }

        intermediate_channel_dict = {
            'mobilenet_v2': {
                'cifar10': 32 * 4 * 4,
                'tinyimagenet': 32 * 8 * 8
            },
            'resnet18': {
                'tinyimagenet': 64 * 8 * 8
            }
        }

        self.projected_size = projected_size_dict[self.dataset_type]
        self.intermediate_channel = intermediate_channel_dict[args.model_type][args.dataset_type]
        self.positive_prototypes = torch.zeros(self.num_classes, self.projected_size).to(self.device)
        self.negative_prototypes = torch.zeros(self.num_classes, self.projected_size).to(self.device)
    
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        self.temperature = 0.5

        # P_SFLなら実行する
        self.sub_projection_heads = {}
        self.sub_optimizers = {}
        self.build_sub_projection_head()
    
    def build_sub_projection_head(self):

        for client_id in range(self.args.num_clients):

            # self.sub_projection_heads[client_id] = nn.Sequential(
            #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2),
            #     nn.Flatten(),
            #     nn.Linear(self.intermediate_channel, self.projected_size)
            # )

            self.sub_projection_heads[client_id] = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU6(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(int(self.intermediate_channel / 4), self.projected_size)
            )

            self.sub_projection_heads[client_id].to(self.device)
        
            self.sub_optimizers[client_id] = torch.optim.SGD(
                params=self.sub_projection_heads[client_id].parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        
    def reset(self):

        self.positive_features = []
        self.positive_labels = []
        self.negative_features = []
        self.negative_labels = []

        self.data_counts = 0
        self.positive_counts = 0
        self.negative_counts = 0

    def save_projected(
        self,
        f_proj: torch.Tensor,
        outs: torch.Tensor,
        labels: torch.Tensor
    ):
        '''
        train datasetを用いた評価の時に一緒に実行される
        ここではとりあえずtrain datasetによるprojection headの出力をpositiveとnegativeに分けて保存していく
        '''
        with torch.no_grad():

            self.data_counts += labels.size(0)
            smax_outs = F.softmax(outs, dim=1)

            for idx, label in enumerate(labels):

                # if smax_outs[idx, label] > self.threshold:
                # 正解クラスの確率がそのクラスの閾値よりも高い場合
                if smax_outs[idx, label] > self.class_threshold[label]:

                    self.positive_features.append(f_proj[idx].unsqueeze(0))
                    self.positive_labels.append(labels[idx].unsqueeze(0))
                    self.positive_counts += 1

                # else:
                # 正解クラスの確率が期待値よりも低い場合
                elif smax_outs[idx, label] < self.threshold:
                    self.negative_features.append(f_proj[idx].unsqueeze(0))
                    self.negative_labels.append(labels[idx].unsqueeze(0))
                    self.negative_counts += 1
                    
    def calculate_prototypes(self):

        # initialize prototype
        previous_positive_prototypes = self.positive_prototypes
        previous_negative_prototypes = self.negative_prototypes

        self.positive_prototypes = torch.zeros(self.num_classes, self.projected_size).to(self.device)
        self.negative_prototypes = torch.zeros(self.num_classes, self.projected_size).to(self.device)

        data_size_per_class = self.data_counts / self.num_classes

        with torch.no_grad():

            if self.positive_features: self.positive_features = torch.cat(self.positive_features, dim=0) 
            if self.negative_features: self.negative_features = torch.cat(self.negative_features, dim=0)
            if self.positive_labels: self.positive_labels = torch.cat(self.positive_labels, dim=0)
            if self.negative_labels: self.negative_labels = torch.cat(self.negative_labels, dim=0)

            # self.flag = True
            self.pos_start = True
            self.neg_start = True
            for cls in range(self.num_classes):

                positive_features_of_cls = self.positive_features[self.positive_labels == cls]
                negative_features_of_cls = self.negative_features[self.negative_labels == cls]

                # データ数の1%以上あれば平均値で更新し、閾値も更新
                if len(positive_features_of_cls) > data_size_per_class * 0.01: # ここの0.01はハイパーパラメータになり得る
                    self.positive_prototypes[cls] = positive_features_of_cls.mean(0)
                    self.class_threshold[cls] += self.args.threshold
                    self.class_threshold[cls] = min(self.class_threshold[cls], 1-self.args.threshold)
                else:
                    self.positive_prototypes[cls] = previous_positive_prototypes[cls]
                    # 1つでもポジティブプロトタイプを計算できないクラスがあったらFalse
                    self.pos_start = False
                    self.class_threshold[cls] -= self.args.threshold
                    self.class_threshold[cls] = max(self.class_threshold[cls], self.args.threshold)
                    print(f'few positive sample in class {cls}!!')

                if len(negative_features_of_cls) > 0:
                    # 既に対照学習が始まっているなら重み付け
                    if self.start:
                        self.negative_prototypes[cls] = (1 - 0.1) * previous_negative_prototypes[cls] + 0.1 * negative_features_of_cls.mean(0)
                    # 今回が初めての場合は、単純に平均値
                    else:
                        self.negative_prototypes[cls] = negative_features_of_cls.mean(0)
                
                else:
                    self.negative_prototypes[cls] = previous_negative_prototypes[cls]
                    # 1つでもネガティブプロトタイプを計算できないクラスがあったらFalse
                    self.neg_start = False
                    print(f'few negative sample in class {cls}!!')
            
            if (self.pos_start and self.neg_start): self.flag = True

            if self.flag:
                self.positive_prototypes = F.normalize(self.positive_prototypes, dim=1)
                self.negative_prototypes = F.normalize(self.negative_prototypes, dim=1)

        print(f'[pos/total]: {self.positive_counts}/{self.data_counts}')
        print(f'[neg/total]: {self.negative_counts}/{self.data_counts}')
        print(f'current threshold of positive or negative: {self.class_threshold}')

    def calculate_loss(
        self,
        client_id: int,
        smashed_data: torch.Tensor,
        labels: torch.Tensor
    ):
        
        self.sub_optimizers[client_id].zero_grad()
        
        projected_smashed_data = self.sub_projection_heads[client_id](smashed_data)
        projected_smashed_data = F.normalize(projected_smashed_data, dim=1)

        positive_sample = self.positive_prototypes[labels]
        negative_sample = self.negative_prototypes[labels]

        pos = self.cosine(projected_smashed_data, positive_sample)
        neg = self.cosine(projected_smashed_data, negative_sample)

        logits = torch.cat((pos.reshape(-1, 1), neg.reshape(-1, 1)), dim=1)
        logits /= 0.5
        logits_labels = torch.zeros(self.batch_size).to(self.device).long()

        loss = self.criterion(logits, logits_labels)

        return loss


    def calculate_loss_pm(
        self,
        client_id: int,
        projected_features: torch.Tensor,
        labels: torch.Tensor
    ):
                
        projected_features = F.normalize(projected_features, dim=1)
        positive_sample = self.positive_prototypes[labels]
        negative_sample = self.negative_prototypes[labels]

        pos = self.cosine(projected_features, positive_sample)
        neg = self.cosine(projected_features, negative_sample)

        logits = torch.cat((pos.reshape(-1, 1), neg.reshape(-1, 1)), dim=1)
        logits /= 0.5
        logits_labels = torch.zeros(self.batch_size).to(self.device).long()

        loss = self.criterion(logits, logits_labels)

        return loss