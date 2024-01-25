import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from torch.autograd import Variable

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout, GAT_Layers, W_size):

        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers1 = nn.ModuleList()
        self.gat_layers2 = nn.ModuleList()
        self.w_h = nn.Linear(in_size, W_size, bias=False)
        self.nums_GAT = GAT_Layers
        if GAT_Layers == 3:
            for i in range(num_meta_paths):
                self.gat_layers.append(GATConv(W_size, 128, 1, dropout, 0, activation=F.elu))
            for i in range(num_meta_paths):
                self.gat_layers1.append(GATConv(128, 128, 1, dropout, 0, activation=F.elu))
            for i in range(num_meta_paths):
                self.gat_layers2.append(GATConv(128, out_size, layer_num_heads, dropout, 0, activation=F.elu))

        if GAT_Layers == 2:
            for i in range(num_meta_paths):
                self.gat_layers.append(GATConv(W_size, 128, 1, dropout, 0, activation=F.elu))
            for i in range(num_meta_paths):
                self.gat_layers1.append(GATConv(128, out_size, layer_num_heads, dropout, 0, activation=F.elu))

        if GAT_Layers == 1:
            for i in range(num_meta_paths):
                self.gat_layers.append(GATConv(W_size, out_size, layer_num_heads, dropout, 0, activation=F.elu))

        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths


    def forward(self, gs, h):
        semantic_embeddings = []

        for i, new_g in enumerate(gs):
            w_h = self.w_h(h)

            if self.nums_GAT == 3:
                semantic_embeddings.append(self.gat_layers2[i](new_g, self.gat_layers1[i](new_g,self.gat_layers[i](new_g,w_h).flatten(1)).flatten(1)).flatten(1))
            if self.nums_GAT == 2:
                semantic_embeddings.append(self.gat_layers1[i](new_g, self.gat_layers[i](new_g, w_h).flatten(1)).flatten(1))
            if self.nums_GAT == 1:
                new_g = dgl.add_self_loop(new_g)
                semantic_embeddings.append(self.gat_layers[i](new_g, w_h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout, GAT_Layers, W_size):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout, GAT_Layers, W_size))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l - 1],hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size, bias=False)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters * 2, kernel_size=k_size, stride=1, padding=k_size // 2),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size // 2),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size // 2),
        )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        output = self.out(x)
        output = output.squeeze()
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2

class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.layer2 = nn.Linear(128, size)

    def forward(self, x, init_dim, num_filters, k_size):
        x = self.layer(x)
        x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))
        x = self.convt(x)
        x = x.permute(0,2,1)
        x = self.layer2(x)
        return x


class HAN_DTI(nn.Module):

    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, num_heads, dropout, GAT_Layers, W_size):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size, out_size, num_heads, dropout, GAT_Layers, W_size))


    def forward(self, s_g, s_h_1, s_h_2,dataset):

        s_g[0] = [g.to("cuda:0") for g in s_g[0]]
        s_g[1] = [g.to("cuda:0") for g in s_g[1]]

        h1_dp = self.sum_layers[0](s_g[0], s_h_1)
        h1_dp = torch.nn.functional.normalize(h1_dp,dim=0)
        h1_dp = torch.nn.functional.normalize(h1_dp, dim=1)

        h2_dp = self.sum_layers[1](s_g[1], s_h_2)
        h2_dp = torch.nn.functional.normalize(h2_dp, dim=0)
        h2_dp = torch.nn.functional.normalize(h2_dp, dim=1)

        h1_nodp = self.sum_layers[0](s_g[0][:-1], s_h_1)
        h2_nodp = self.sum_layers[1](s_g[1][:-1], s_h_2)
        h1_nodp = torch.nn.functional.normalize(h1_nodp, dim=0)
        h2_nodp = torch.nn.functional.normalize(h2_nodp, dim=0)
        h1_nodp = torch.nn.functional.normalize(h1_nodp, dim=1)
        h2_nodp = torch.nn.functional.normalize(h2_nodp, dim=1)

        if dataset=='luo':
            h1 = h1_dp * 0.2 + h1_nodp * 0.8
            h2 = h2_dp * 0.2 + h2_nodp * 0.8
        elif dataset=='zheng':
            h1 = h1_dp * 0.7 + h1_nodp * 0.3
            h2 = h2_dp * 0.7 + h2_nodp * 0.3

        return h1, h2, torch.matmul(h1, h2.t())
