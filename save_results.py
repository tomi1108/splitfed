import os
import csv
import argparse
from datetime import datetime

def results_setting(
    args: argparse.ArgumentParser
):
    
    results_dir = args.results_dir
    dataset_name = args.dataset_type
    model_name = args.model_type
    approach_name = args.app_name
    current_date = datetime.now().strftime('%Y-%m-%d')

    results_dir = os.path.join(results_dir, dataset_name, model_name, approach_name, current_date)

    if not os.path.exists(results_dir):

        os.makedirs(results_dir)
        print(f'Directory {results_dir} is created.')
    
    rounds = 'R' + str(args.num_rounds)
    epochs = 'E' + str(args.num_epochs)
    batch  = 'B' + str(args.batch_size)

    if args.data_dist_type == 'iid':
        dist_type = 'iid'
    elif args.data_dist_type == 'non-iid':
        dist_type = 'A' + str(args.alpha)

    if approach_name == 'P_SFL' or approach_name == 'PKL_SFL':
        dist_type += '_U' + str(args.mu) + '_L' + str(args.lam)


    filename = '_'.join([rounds, epochs, batch, f'C{str(args.num_clients)}', dist_type]) + '.csv'
    results_dir = os.path.join(results_dir, filename)
    header = ['round', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']

    with open(results_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    return results_dir


def save_data(
    path: str,
    write_data: list
):
    
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(write_data)