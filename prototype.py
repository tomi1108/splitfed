import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm


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


def prGreen(skk):
    print("\033[92m {}\033[00m" .format(skk))


class Sub_Classifier(nn.Module):

    def __init__(self, input_size, projected_size, num_classes):
        super(Sub_Classifier, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, projected_size)
        self.bn = nn.BatchNorm1d(projected_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(projected_size, num_classes)
    
    def forward(self, x):

        projected = self.linear1(self.flatten(x))
        output = self.linear2(self.relu(self.bn(projected)))

        return projected, output


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

        # self.threshold = 1 / num_classes # 期待値として定義
        self.threshold = min(0.1, 1 / num_classes**0.5)
        self.lam = args.lam
        self.batch_size = args.batch_size
        self.dataset_type = args.dataset_type

        # クラスごとの閾値のリスト
        self.class_threshold = [ self.threshold for _ in range(self.num_classes) ]
        self.negative_class_threshold = [ self.threshold for _ in range(self.num_classes) ]
        
        self.positive_features = []
        self.positive_labels = []
        self.negative_features = []
        self.negative_labels = []

        self.data_counts = 0
        self.positive_counts = 0
        self.negative_counts = 0
        self.flag = False

        self.pos_flag = [ False for _ in range(num_classes) ]
        self.neg_flag = [ False for _ in range(num_classes) ]

        self.pos_start = False
        self.neg_start = False

        self.projected_size = args.projected_size
        self.intermediate_info = get_intermediate_info(args)
        self.positive_prototypes = torch.zeros(self.num_classes, self.projected_size).to(self.device)
        self.negative_prototypes = torch.zeros(self.num_classes, self.projected_size).to(self.device)
    
        self.criterion = torch.nn.CrossEntropyLoss()
        self.kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        self.temperature = 0.5

        app_name = self.args.app_name
        if app_name == 'P_SFL':
            self.sub_projection_heads, self.sub_optimizers, self.sub_schedulers = self.build_sub_projection_head()
        elif app_name == 'PKL_SFL':
            self.sub_classifier, self.sub_optimizer, self.sub_scheduler = self.build_sub_classifier()

    def build_sub_classifier(self):

        in_c, h, w = self.intermediate_info
        input_size = in_c * h * w
        
        sub_classifier = Sub_Classifier(input_size, self.projected_size, self.num_classes).to(self.device)

        sub_optimizer = torch.optim.SGD(
            params = sub_classifier.parameters(),
            lr = self.args.lr,
            momentum = self.args.momentum,
            weight_decay = self.args.weight_decay
        )

        sub_scheduler = CosineAnnealingLR(
            optimizer = sub_optimizer,
            T_max = self.args.num_rounds - self.args.warmup_rounds,
            eta_min = self.args.min_lr,
            last_epoch = -1
        )

        return sub_classifier, sub_optimizer, sub_scheduler
    
    def build_sub_projection_head(self):

        in_c, h, w = self.intermediate_info
        out_c = int(in_c / 4)
        sub_projection_heads, sub_optimizers, sub_schedulers = {}, {}, {}

        for client_id in range(self.args.num_clients):

            sub_projection_heads[client_id] = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(int(out_c * h * w / 2**2), self.projected_size)
            )

            sub_projection_heads[client_id].to(self.device)
        
            sub_optimizers[client_id] = torch.optim.SGD(
                params=sub_projection_heads[client_id].parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

            sub_schedulers[client_id] = CosineAnnealingLR(
                optimizer = sub_optimizers[client_id],
                T_max = self.args.num_rounds - self.args.warmup_rounds,
                eta_min = self.args.min_lr,
                last_epoch = -1
            )

        return sub_projection_heads, sub_optimizers, sub_schedulers
    
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

                # 正解クラスの確率がそのクラスの閾値よりも高い場合
                if smax_outs[idx, label] > self.class_threshold[label]:

                    self.positive_features.append(f_proj[idx].unsqueeze(0))
                    self.positive_labels.append(labels[idx].unsqueeze(0))
                    self.positive_counts += 1

                # 正解クラスの確率が期待値よりも低い場合
                elif smax_outs[idx, label] < self.negative_class_threshold[label]:
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
                    self.class_threshold[cls] = min(self.class_threshold[cls]+self.threshold, 1-self.threshold)
                    if not self.pos_flag[cls]: self.pos_flag[cls] = True
                else:
                    self.positive_prototypes[cls] = previous_positive_prototypes[cls]
                    # 1つでもポジティブプロトタイプを計算できないクラスがあったらFalse
                    self.pos_start = False
                    self.class_threshold[cls] = max(self.class_threshold[cls]-self.threshold, self.threshold)
                    print(f'few positive sample in class {cls}!!')

                if len(negative_features_of_cls) > 0:
                    # 既に対照学習が始まっているなら重み付け
                    if self.neg_flag[cls]:
                        self.negative_prototypes[cls] = (1 - self.lam) * previous_negative_prototypes[cls] + self.lam * negative_features_of_cls.mean(0)
                    else:
                        self.negative_prototypes[cls] = negative_features_of_cls.mean(0)
                        self.neg_flag[cls] = True
                
                else:
                    self.negative_prototypes[cls] = previous_negative_prototypes[cls]
                    # 1つでもネガティブプロトタイプを計算できないクラスがあったらFalse
                    self.neg_start = False
                    print(f'few negative sample in class {cls}!!')
                
                self.negative_class_threshold[cls] = (self.threshold + self.class_threshold[cls]) / 2

            if not self.flag:
                if (self.pos_start and self.neg_start):
                    self.flag = True
                    prGreen("start contrastive learning using prototype!!")

            if self.flag:
                self.positive_prototypes = F.normalize(self.positive_prototypes, dim=1)
                self.negative_prototypes = F.normalize(self.negative_prototypes, dim=1)

        print(f'[pos/total]: {self.positive_counts}/{self.data_counts}')
        print(f'[neg/total]: {self.negative_counts}/{self.data_counts}')
        print(f'current positive threshold: {self.class_threshold}')
        print(f'current negative threshold: {self.negative_class_threshold}')

    def calculate_pkl_loss(
        self,
        smashed_data: torch.Tensor,
        labels: torch.Tensor,
        soft_targets: torch.Tensor,
        p_soft_targets: torch.Tensor
    ):
        
        self.sub_optimizer.zero_grad()
        projected, outputs = self.sub_classifier(smashed_data)

        projected = F.normalize(projected, dim=1)
        soft_projected = F.normalize(p_soft_targets, dim=1)

        positive_sample = self.positive_prototypes[labels]
        negative_sample = self.negative_prototypes[labels]

        pos = self.cosine(projected, positive_sample)
        neg = self.cosine(projected, negative_sample)

        soft_pos = self.cosine(soft_projected, positive_sample)
        soft_neg = self.cosine(soft_projected, negative_sample)

        logits = torch.cat((pos.reshape(-1, 1), neg.reshape(-1, 1)), dim=1)
        logits /= 0.5

        soft_logits = torch.cat((soft_pos.reshape(-1, 1), soft_neg.reshape(-1, 1)), dim=1)
        soft_logits /= 0.5

        logits_labels = torch.zeros(self.batch_size).to(self.device).long()

        # prototype contrastive learning
        # proto_loss = self.criterion(logits, logits_labels)
        proto_loss = self.criterion(logits, logits_labels) + self.criterion(soft_logits, logits_labels)

        # knowledge distillation
        # correct_class_indices = labels.view(-1, 1)
        # correct_class_probs = torch.gather(soft_targets, dim=1, index=correct_class_indices).squeeze()
        # class_thresholds_tensor = torch.tensor([self.class_threshold[label] for label in labels], device=self.device)
        # distillation_mask = (correct_class_probs > class_thresholds_tensor[correct_class_indices]).float()
        
        kl_loss = self.kl_criterion(F.log_softmax(outputs / 2.0, dim=1),
                                    F.softmax(soft_targets / 2.0, dim=1)) * 2.0 * 2.0
        # kl_loss = (kl_loss * distillation_mask).mean()

        return proto_loss, kl_loss

    def calculate_p_loss(
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

        # prototypical contrastive learning
        p_loss = self.criterion(logits, logits_labels)

        return p_loss
