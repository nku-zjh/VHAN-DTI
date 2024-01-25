# -*- coding:utf-8 -*-
"""
作者：zjh
日期：2023年12月03日
"""
import numpy as np

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

datasets = ['drug','protein']  #drug   protein

for dataset in datasets:
    XD = []
    if dataset =='drug':
        filename = 'drug_smiles.txt'
    elif dataset =='protein':
        filename = 'protein_seq.txt'

    with open(filename, 'r') as file:
        lines = file.readlines()

        if dataset=="drug":
            MAX=110
            DICT = CHARISOSMISET
            path = 'drug_SmilesToNumber.txt'
        else:
            MAX = 1150
            DICT = CHARPROTSET
            path = 'protein_SeqToNumber.txt'

        for line in lines:
            line = line[0:-1]
            X = np.zeros(MAX, dtype=np.int64)
            for i, ch in enumerate(line[:MAX]):
                X[i] = DICT[ch]
            XD.append(X)
        # XD_init = Variable(XD.long())
    print(len(XD))

    np.savetxt(path,XD,fmt="%d")

