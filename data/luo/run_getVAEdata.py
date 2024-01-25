# -*- coding:utf-8 -*-
"""
作者：zjh
日期：2023年12月27日
"""
# -*- coding:utf-8 -*-
from model_getVAEdata import net
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np


class MyClass:

    def __init__(self, charsmiset_size, charseqset_size,max_smi_len,max_seq_len):
        self.charsmiset_size = charsmiset_size
        self.charseqset_size = charseqset_size
        self.max_smi_len = max_smi_len
        self.max_seq_len = max_seq_len

    @classmethod
    def print_class_variable(cls):
        print(cls.class_variable)

    def print_info(self):
        print(f"Name: {self.name}, Age: {self.age}")

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        if isinstance(m, nn.LSTM):
            init.orthogonal_(m.all_weights[0][0])
            init.orthogonal_(m.all_weights[0][1])
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)

def loss_f(recon_x, x, mu, logvar):

    cit = nn.CrossEntropyLoss(reduction='none')
    cr_loss = torch.sum(cit(recon_x.permute(0, 2, 1), x), 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return torch.mean(cr_loss + KLD)


epoch=600
dd =['drug','protein'] #drug protein

for d in dd:
    NUM_FILTERS = 100
    embedding_dim = 128

    if d == 'drug':
        max_smi_len = 100
        filename = 'drug_SmilesToNumber.txt'
        charsmiset_size = 64
        drug_kernel_size = 5
    else:
        max_smi_len = 1000
        filename = 'protein_SeqToNumber.txt'
        charsmiset_size = 25
        drug_kernel_size = 7


    FLAGS = MyClass(charsmiset_size=charsmiset_size,charseqset_size=111,max_smi_len=max_smi_len,max_seq_len=111)
    model = net(FLAGS, NUM_FILTERS, drug_kernel_size, 111,embedding_dim).cuda()
    model.apply(weights_init)

    drug_SMILES = np.loadtxt(filename)
    optimizer = optim.Adam(model.parameters())
    drug_SMILES = torch.Tensor(drug_SMILES)

    loss = 99999999
    for i in range(0,epoch):
        model.train()
        loss_func = nn.MSELoss()

        optimizer.zero_grad()

        x_reparametrize,recon_x, drug, mu_drug, logvar_drug = model(drug_SMILES, FLAGS, NUM_FILTERS,drug_kernel_size)
        loss_drug = loss_f(recon_x, drug, mu_drug, logvar_drug)
        total_loss = loss_drug

        print(i,loss_drug)
        print(x_reparametrize.shape)
        if loss > total_loss:
            loss = total_loss
            x_final = x_reparametrize

        total_loss.backward()
        optimizer.step()

    np.savetxt(f'VAEdata/{d}_{3*NUM_FILTERS}_{embedding_dim}_{drug_kernel_size}.txt',x_final.cpu().detach().numpy(),fmt="%.5f")
