import argparse
import numpy as np
from typing import Tuple, List

def get_traindata_info(
    args: argparse.ArgumentParser
) -> Tuple[int, List[int]]:
    
    '''
    データセットのクラス数と、各クラスのデータ数の情報を取得
    今後実装していくデータセットの情報はここに追加する
    '''

    dataset_type = args.dataset_type
    
    dataset_info = {
        'cifar10': {'num_classes': 10, 'data_size_per_class': [5000] * 10},
        'cifar100': {'num_classes': 100, 'data_size_per_class': [500] * 100},
        'tinyimagenet': {'num_classes': 200, 'data_size_per_class': [500] * 200}
    }

    num_classes = dataset_info[dataset_type]['num_classes']
    data_size_per_class = dataset_info[dataset_type]['data_size_per_class']
    
    return num_classes, data_size_per_class


def get_data_indices_per_client(
    args: argparse.ArgumentParser
) -> Tuple[int, List[int], List[List[int]]]:
    
    '''
    num_classes: クラス数
    data_size_per_client: 各クライアントが持つデータ数が格納されたリスト
    data_indices_per_client: 各クライアントが持つデータの、各クラスにおける使用するデータのインデックスが格納されたリスト
    '''

    num_clients = args.num_clients    
    num_classes, data_size_per_class = get_traindata_info(args)
    
    if args.data_dist_type == 'iid':

        data_dist = np.full((num_classes, num_clients), 1 / num_clients)

    elif args.data_dist_type == 'non-iid':

        seed = args.seed
        alpha = args.alpha
        alpha = np.asarray(alpha)
        alpha = np.repeat(alpha, args.num_clients)

        data_dist = np.random.default_rng(seed).dirichlet(
            alpha=alpha,
            size=num_classes
        )

    data_size_per_client = [0] * num_clients
    data_indices_per_client = [ [] for _ in range(num_clients)]

    for cls in range(num_classes):

        data_size = data_size_per_class[cls]
        ptr = 0

        for clt in range(num_clients):

            if clt + 1 < num_clients:

                reduced_data_size = int(data_size * data_dist[cls][clt])
                data_indices_per_client[clt].append(list(range(ptr, ptr + reduced_data_size)))
                ptr += reduced_data_size
            
            elif clt + 1 == num_clients:

                reduced_data_size = int(data_size - ptr)
                data_indices_per_client[clt].append(list(range(ptr, ptr + reduced_data_size)))
                ptr += reduced_data_size

                if ptr != data_size:
                    raise ValueError(f'Data size mismatch: assigned {ptr} indices, but expected {data_size}')
        
            data_size_per_client[clt] += reduced_data_size

    return num_classes, data_size_per_client, data_indices_per_client