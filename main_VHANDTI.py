import torch
from utils_VHANDTI import load_data
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
from sklearn.model_selection import train_test_split
from model_VHANDTI import HAN_DTI


class DTI_PU_loss(nn.Module):
    def __init__(self):
        super(DTI_PU_loss, self).__init__()

    def forward(self, drug_protein_reconstruct, drug_protein, pos_x_index, pos_y_index, neg_x_index,neg_y_index, alpha):
        alpha = alpha
        loss_fn = torch.nn.MSELoss(reduction='none')
        loss_mat = loss_fn(drug_protein_reconstruct, drug_protein)
        loss = (loss_mat[pos_x_index, pos_y_index].sum() * ((1 - alpha) / 2) + loss_mat[neg_x_index, neg_y_index].sum() * (alpha / 2))
        return loss


def evaluate(model, g, features_d, features_p, DTItest, DTIvalid,dataset):
    model.eval()
    pred_list = []
    ground_truth = []
    with torch.no_grad():
        d_x, p_x, logits = model(g, features_d, features_p,dataset)
        logits = logits.cpu().numpy()

    DTIvalid = DTIvalid.cpu().numpy()
    DTItest = DTItest.cpu().numpy()

    for ele in DTIvalid:
        pred_list.append(logits[ele[0], ele[1]])
        ground_truth.append(ele[2])
    valid_auc = roc_auc_score(ground_truth, pred_list)
    valid_aupr = average_precision_score(ground_truth, pred_list)

    pred_list = []
    ground_truth = []
    for ele in DTItest:
        pred_list.append(logits[ele[0], ele[1]])
        ground_truth.append(ele[2])
    test_auc = roc_auc_score(ground_truth, pred_list)
    fpr, tpr, thresholds = roc_curve(ground_truth, pred_list)
    test_aupr = average_precision_score(ground_truth, pred_list)
    precision, recall, thresholds = precision_recall_curve(ground_truth, pred_list)

    return valid_auc, valid_aupr, test_auc, test_aupr, fpr, tpr, precision, recall

def train_and_evaluate(i, DTItrain, DTIvalid, DTItest, graph, pos_x_index, pos_y_index, neg_x_index, neg_y_index,
                       drug_protein_train, train_mask, features_d, features_p, epochs, in_size, out_size, loss_alpha,dataset):
    pos_x_index = torch.tensor(pos_x_index, dtype=torch.long)
    pos_y_index = torch.tensor(pos_y_index, dtype=torch.long)
    neg_x_index = torch.tensor(neg_x_index, dtype=torch.long)
    neg_y_index = torch.tensor(neg_y_index, dtype=torch.long)
    DTItrain = torch.from_numpy(DTItrain).long()
    DTIvalid = torch.from_numpy(DTIvalid).long()
    DTItest = torch.from_numpy(DTItest).long()
    drug_protein_train = torch.from_numpy(drug_protein_train).float()
    train_mask = torch.from_numpy(train_mask).float()
    model = HAN_DTI(
        all_meta_paths=[len(graph[0]), len(graph[1])],
        in_size=in_size,
        hidden_size=args['hidden_units'],
        out_size=args['out_size'],
        num_heads=args['num_heads'],
        dropout=args['dropout'],
        GAT_Layers=args['Gat_layers'],
        W_size=args['W_size']).to(args['device'])

    loss_fcn = DTI_PU_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    DTItrain = DTItrain.to(args['device'])
    DTIvalid = DTIvalid.to(args['device'])
    DTItest = DTItest.to(args['device'])
    train_mask = train_mask.to(args['device'])
    drug_protein = drug_protein_train.to(args['device'])
    pos_x_index = pos_x_index.to(args['device'])
    pos_y_index = pos_y_index.to(args['device'])
    neg_x_index = neg_x_index.to(args['device'])
    neg_y_index = neg_y_index.to(args['device'])

    best_valid_auc = 0
    best_valid_aupr = 0
    best_test_auc = 0
    best_test_aupr = 0
    best_loss = 0
    model.train()
    for epoch in range(epochs):
        if epoch % 300 == 0:
            optimizer.param_groups[0]['lr'] *= 0.9

        optimizer.zero_grad()
        d, p, logits = model(graph, features_d, features_p,dataset)
        loss_pre = loss_fcn(logits, drug_protein, pos_x_index, pos_y_index, neg_x_index, neg_y_index,loss_alpha)
        loss_pre.backward()
        optimizer.step()

        valid_auc, valid_aupr, test_auc_0, test_aupr_0, fpr, tpr, precision, recall = evaluate(model, graph, features_d,features_p, DTItest,DTIvalid,dataset)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_valid_aupr = valid_aupr

        if test_auc_0 > best_test_auc:
            best_test_auc = test_auc_0
            best_test_aupr = test_aupr_0
            best_loss = loss_pre.item()

        print('Epoch {:d} | Train Loss {:.4f} | best_valid_auc {:.4f} | best_valid_aupr {:.4f} |'
              'test_auc {:.4f} |test_aupr {:.4f} |best_test_auc {:.4f} |best_test_aupr {:.4f}'.format(
            epoch + 1, loss_pre.item(), best_valid_auc, best_valid_aupr, test_auc_0, test_aupr_0, best_test_auc,best_test_aupr))
    return best_loss,best_valid_auc, best_valid_aupr, best_test_auc, best_test_aupr, fpr, tpr, precision, recall


def get_train(DTItrain, num_drug, num_protein):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    # print (DTItrain)
    pos_x_index = []
    pos_y_index = []
    neg_x_index = []
    neg_y_index = []
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
        if ele[2] == 1:
            pos_x_index.append(ele[0])
            pos_y_index.append(ele[1])
        if ele[2] == 0:
            neg_x_index.append(ele[0])
            neg_y_index.append(ele[1])

    train_mask = mask
    return pos_x_index, pos_y_index, neg_x_index, neg_y_index, drug_protein, train_mask


def main(args):
    alpha = args['alpha']
    out_size = args['out_size']
    dataset = args['dataset']
    #need to change
    hd = torch.FloatTensor(np.loadtxt(f'data/{dataset}/VAEdata/drug_384_128_4.txt'))
    hp = torch.FloatTensor(np.loadtxt(f'data/{dataset}/VAEdata/protein_384_128_8.txt'))

    in_size = [hd.shape[1], hp.shape[1]]
    test_auc_round = []
    test_aupr_round = []
    # best_loss_round = []

    test_auc_fold = []
    test_aupr_fold = []
    test_loss_fold = []

    for i in range(0, 10):
        features_d = hd.to(args['device'])
        features_p = hp.to(args['device'])

        # rs = args['seed']
        DTItest, DTItrain, graph, num_drug, num_protein = load_data(i, args['dataset'], args['ratio'],args['network_path'])
        DTItrain, DTIvalid = train_test_split(DTItrain, test_size=0.05, random_state=2)
        print("#############%d fold" % i + "#############")
        # fold = fold + 1
        pos_x_index, pos_y_index, neg_x_index, neg_y_index, drug_protein_train, train_mask = get_train(DTItrain,num_drug,num_protein)
        best_loss,best_valid_auc, best_valid_aupr, best_test_auc, best_test_aupr, fpr, tpr, precision, recall = train_and_evaluate(
            i, DTItrain, DTIvalid, DTItest, graph, pos_x_index, pos_y_index, neg_x_index, neg_y_index,
            drug_protein_train, train_mask, features_d, features_p, args['num_epochs'], in_size, out_size, alpha,dataset)
        test_auc_fold.append(best_test_auc)
        test_aupr_fold.append(best_test_aupr)
        test_loss_fold.append(best_loss)
        # p = f'{i}:'
        test_auc_round.append(f'{i}：{best_test_auc}')
        test_aupr_round.append(f'{i}：{best_test_aupr}')
        # best_loss_round.append(f'{i}：{best_loss}')
    test_auc_round.append(f'a：{np.mean(test_auc_fold)}')
    test_aupr_round.append(f'a：{np.mean(test_aupr_fold)}')
    # best_loss_round.append(f'a：{np.mean(test_loss_fold)}')

    merged_list = [f"{item1} {item2}" for item1, item2 in zip(test_auc_round, test_aupr_round)]
    np.savetxt(args['log_dir'] + f'/test_auc_aupr_loss.txt', merged_list, fmt='%s')


if __name__ == '__main__':
    import argparse
    from utils_VHANDTI import setup
    dataset = 'zheng' # luo zheng
    print(dataset,"dataset")
    pos_neg = 1 # 1  , 5  , 10 mean 1:1,1:5,1:10
    parser = argparse.ArgumentParser('VHAN-DTI')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default=f'results{dataset}',
                        help='Dir for saving training results')
    parser.add_argument('-data', '--data', type=str, default=f'{dataset}',
                        help='different dataset. eg.hetero,Es,ICs,GPCRs,Ns ')
    parser.add_argument('-r', '--ratio', type=str, default=f'{pos_neg}',
                        help='posive :negative 1,5,10')
    parser.add_argument('-path', '--network_path', type=str, default=f'data/{dataset}/',
                        help='different dataset path.eg.hetero_data,Es,ICs,GPCRs,Ns')

    args = parser.parse_args().__dict__
    args = setup(args)
    # print(args)
    main(args)