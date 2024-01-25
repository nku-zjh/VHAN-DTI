# -*- coding:utf-8 -*-
"""
作者：zjh
日期：2023年10月17日
"""
import  numpy as np
import os

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def heterToHomo(hetero_data):
    x = np.array(hetero_data)
    y = np.dot(x, x.T)
    y[y > 0] = 1
    np.fill_diagonal(y, 1)
    # print("内积是否为方阵：", np.array_equal(y, y.T))

    return  y


pos_neg = 1 # 1  , 5  , 10 mean 1:1,1:5,1:10
create_folder(f'pos_neg_1_{pos_neg}/tenFolds/')
for i in range(10):
    DTItrain = []
    DTItest = np.loadtxt(f'pos_neg_1_{pos_neg}/dp_index_0{i}.txt', dtype=int)
    for j in range(10):
        if j != i:
            DTItrain.append(np.loadtxt(f'pos_neg_1_{pos_neg}/dp_index_0{j}.txt', dtype=int))
    DTItrain = np.concatenate(DTItrain, axis=0)

    filename_test = f'pos_neg_1_{pos_neg}/tenFolds/dp_test_{i:02d}.txt'
    with open(filename_test, 'w') as file:
        for triplet in DTItest:
            file.write(f'{triplet[0]} {triplet[1]} {triplet[2]}\n')

    dp_heter = np.zeros((1094, 1556))
    # pd_heter = np.zeros((1512, 708))
    filename_train = f'pos_neg_1_{pos_neg}/tenFolds/dp_train_{i:02d}.txt'
    filename_dp_homo = f'pos_neg_1_{pos_neg}/tenFolds/dp_homo_{i:02d}.txt'
    filename_pd_homo = f'pos_neg_1_{pos_neg}/tenFolds/pd_homo_{i:02d}.txt'
    with open(filename_train, 'w') as file:
        for triplet in DTItrain:
            dp_heter[triplet[0]][triplet[1]]=triplet[2]
            file.write(f'{triplet[0]} {triplet[1]} {triplet[2]}\n')
        pd_heter = dp_heter.T
        dp_homo = heterToHomo(dp_heter)
        pd_homo = heterToHomo(pd_heter)
        np.savetxt(filename_dp_homo, dp_homo, "%d")
        np.savetxt(filename_pd_homo, pd_homo, "%d")
        # np.savetxt(f'pos_neg_1_{pos_neg}/tenFolds/dp_heter_{i:02d}.txt', dp_heter, "%d")
        # np.savetxt(f'pos_neg_1_{pos_neg}/tenFolds/pd_heter_{i:02d}.txt', pd_heter, "%d")
