import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import scipy.sparse as sp


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix

def setup_log_dir(args, sampling=False):
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))
    if sampling:
        log_dir = log_dir + '_sampling'
    mkdir_p(log_dir)
    return log_dir

default_configure = {
    'lr': 0.001,
    'num_heads': [8],  # Number of attention heads
    'hidden_units': 8,
    'dropout': 0.4,
    'weight_decay': 0.01,
    'num_epochs': 10,#luo 1500 zheng 3000
    'W_size': 128,
    'out_size': 64,
    'Gat_layers': 1,
    'alpha': 0.6
}

sampling_configure = {
    'batch_size': 20
}


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    if args['data'] == 'luo':
        args['dataset'] = 'luo'
    elif args['data'] == 'zheng':
        args['dataset'] = 'zheng'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def load_homo(i,dataset,network_path, r):
    network_path1 = f"{network_path}homo_net/"
    r=int(r)
    if r==1:
        # 1:1
        network_path2 = f"{network_path}pos_neg_1_1/tenFolds/"
    elif r==5:
        # 1:5
        network_path2 = f"{network_path}pos_neg_1_5/tenFolds/"
    elif r==10:
        # 1:10
        network_path2 = f"{network_path}pos_neg_1_10/tenFolds/"
    if dataset=='luo':
        num_drug = 708
        num_protein = 1512
    elif dataset=='zheng':
        num_drug = 1094
        num_protein = 1556

    DTItest = np.loadtxt(network_path2 + f'dp_test_0{i}.txt', dtype=int)
    DTItrain = np.loadtxt(network_path2 + f'dp_train_0{i}.txt', dtype=int)

    drug_drug = np.loadtxt(network_path1 + 'mat_drug_drug.txt')
    # First [0:708] are drugs, the rest are compounds retrieved from ZINC15 database
    drug_chemical = np.loadtxt(network_path1 + 'Similarity_Matrix_Drugs.txt')
    if dataset=='luo':
        drug_chemical = drug_chemical[:708, :708]
    drug_protein = np.loadtxt(network_path2 + f'dp_homo_0{i}.txt', dtype=int)

    #drug
    d_d = dgl.from_scipy(sp.csr_matrix(drug_drug))
    d_c = dgl.from_scipy(sp.csr_matrix(drug_chemical))
    d_p = dgl.from_scipy(sp.csr_matrix(drug_protein))


    protein_protein = np.loadtxt(network_path1 + 'mat_protein_protein.txt')
    protein_sequence = np.loadtxt(network_path1 + 'Similarity_Matrix_Proteins.txt')
    protein_drug = np.loadtxt(network_path2 + f'pd_homo_0{i}.txt', dtype=int)

    #protein
    p_p = dgl.from_scipy(sp.csr_matrix(protein_protein))
    p_s = dgl.from_scipy(sp.csr_matrix(protein_sequence))
    p_d = dgl.from_scipy(sp.csr_matrix(protein_drug))

    graph_drug = [ d_d, d_c,d_p]
    graph_protein = [p_p, p_s,p_d]

    graph = [graph_drug,graph_protein]

    print('homo dataset loaded')
    return DTItest, DTItrain,graph, num_drug, num_protein


def load_data(i,dataset, r, network_path):
    if dataset == 'luo' or dataset == 'zheng':
        return load_homo(i,dataset,network_path, r)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))



