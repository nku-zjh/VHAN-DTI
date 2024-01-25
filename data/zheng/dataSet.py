# -*- coding:utf-8 -*-
"""
作者：zjh
日期：2023年10月17日
"""
import numpy as np
import os

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")





drug_protein = np.loadtxt('mat_drug_protein.txt')

row_indices, col_indices = np.where(drug_protein == 1)
row_indices_0, col_indices_0 = np.where(drug_protein == 0)
random_order = np.random.permutation(len(row_indices))
random_order_0 = np.random.permutation(len(row_indices_0))


num_folds = 10
fold_size = len(row_indices) // num_folds
folds = []
pos_neg = 1 # 1  , 5  , 10 mean 1:1,1:5,1:10
for i in range(num_folds):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(row_indices)
    fold_row_indices = row_indices[random_order[start_idx:end_idx]]
    fold_col_indices = col_indices[random_order[start_idx:end_idx]]
    fold_values = np.ones_like(fold_row_indices)
    fold_triplets = list(zip(fold_row_indices, fold_col_indices, fold_values))

    if i==9:
        fold_size_0 = 1181*pos_neg
    else:
        fold_size_0 = len(fold_triplets) * pos_neg
    start_idx_0 = i * fold_size_0
    end_idx_0 = (i + 1) * fold_size_0 if i < num_folds - 1 else len(row_indices)*pos_neg
    fold_row_indices_0 = row_indices_0[random_order_0[start_idx_0:end_idx_0]]
    fold_col_indices_0 = col_indices_0[random_order_0[start_idx_0:end_idx_0]]
    fold_values_0 = np.zeros_like(fold_row_indices_0)
    fold_triplets_0 = list(zip(fold_row_indices_0, fold_col_indices_0, fold_values_0))
    folds.append(fold_triplets+fold_triplets_0)


folder_path = f'pos_neg_1_{pos_neg}'

create_folder(folder_path)

for i, fold in enumerate(folds):
    filename = f'pos_neg_1_{pos_neg}/dp_index_{i:02d}.txt'
    with open(filename, 'w') as file:
        for triplet in fold:
            file.write(f'{triplet[0]} {triplet[1]} {triplet[2]}\n')