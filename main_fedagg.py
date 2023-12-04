import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import argparse
import logging
import os
import numpy as np
import torch
from model_zoo import create_model
import argparse
from fedagg import run_fedagg
from model_zoo import create_model
from data_loader import load_partition_data_cifar10
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def add_args(parser):
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--comm_round', type=int, default=1000, metavar='N',
                        help='maximum communication rounds (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--client_number', type=int, default=225, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--edge_number', type=int, default=15,
                        help='number of edges')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=3.0, metavar='PA',
                        help='partition alpha (default: 3.0)')
    parser.add_argument('--method', type=str, default='fedagg',
                        help='method_name')
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--class_num', type=int, default=10,
                        help='class_num')
    parser.add_argument("--T_agg", nargs="*", type=float, default=3.0, help="T_agg")

    args = parser.parse_args()
    args.client_number_per_round=args.client_number
    args.personal_learning_rate=args.lr
    args.client_num_in_total=args.client_number
    return args


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        data_loader = load_partition_data_cifar10
    else:
        raise Exception("dataset not implemented error")

    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num_train, class_num_test = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_number, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test]
    return dataset


def create_edge_model(args, n_classes,index):
    return create_model("resnet10")

def create_client_model(args, n_classes,index):
    return create_model("cnn")
    

def create_client_models(args, n_classes):
    random.seed(123)
    client_models=[]
    for _ in range(args.client_number):
        client_models.append(create_client_model(args,n_classes,_))
    return client_models

def create_edge_models(args, n_classes):
    random.seed(456)
    edge_models=[]
    for _ in range(args.edge_number):
        edge_models.append(create_edge_model(args,n_classes,_))
    return edge_models


def create_cloud_model(args, n_classes):
    return create_model("resnet18")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(5))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, test_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num_train, class_num_test] = dataset
    if args.method == 'fedagg':
        client_models=create_client_models(args,class_num_train)
        edge_models=create_edge_models(args,class_num_train)
        cloud_model=create_cloud_model(args,class_num_train)
        run_fedagg(client_models,edge_models,cloud_model,train_data_local_num_dict, test_data_local_num_dict,train_data_local_dict, test_data_local_dict, test_data_global, args)
    else:
        raise Exception("method not implemented error")


