from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def cmd(x1, x2):
    n_moments = 5
    def matchnorm(x1, x2):
        return ((x1 - x2)**2).sum().sqrt()
    def scm(sx1, sx2, k):
        ss1 = (sx1**k).mean(0)
        ss2 = (sx2**k).mean(0)
        return matchnorm(ss1, ss2)
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = matchnorm(mx1, mx2)
    scms = dm
    for i in range(n_moments - 1):
        scms = scms + scm(sx1, sx2, i+2)
    return scms


def metapath(types, relations, source, target, hops=2):
    final_paths = []
    current_paths = [[source]]
    for _ in range(hops):
        next_paths = []
        for current_path in current_paths:
            if len(current_path) == 1:
                target_t = current_path[0]
            else:
                target_t = relations[current_path[-1]]['source']
            for r in types[target_t]:
                new_path = current_path.copy()
                new_path.append(r)
                source_t = relations[r]['source']
                if source_t == target:
                    final_paths.append(new_path)
                else:
                    next_paths.append(new_path)
        current_paths = next_paths
    #print("Meta Paths: ", final_paths)
    return final_paths


class Classifier(nn.Module):
    def __init__(self, n_in, n_out, ranking=False):
        super(Classifier, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.ranking = ranking
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        nn.init.xavier_uniform_(self.linear.weight)

    def get_parameters(self):
        ml = list()
        ml.append({'params': self.linear.parameters()})
        return ml

    def forward(self, x):
        y = self.linear(x)
        return torch.log_softmax(y, dim=-1)

    def calc_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)

    def calc_acc(self, y_pred, y_true):
        if self.ranking:
            test_res = []
            test_ndcg = []
            for ai, bi in zip(y_true, torch.argsort(y_pred, dim=-1, descending=True)):
                resi = ai[bi].cpu().numpy()
                test_res += [resi]
                test_ndcg += [ndcg_at_k(resi, len(resi))]
            test_ndcg = np.average(test_ndcg)
            test_mrr = np.average(mean_reciprocal_rank(test_res))
            return test_ndcg, test_mrr
        else:
            y_pred = torch.argmax(y_pred, dim=1).cpu()
            y_true = torch.argmax(y_true, dim=1).cpu()
            return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


class GNN(nn.Module):
    def __init__(self, device, source_node, target_node, use_matching_loss, matching_hops, \
                 types, relations, in_dims, hid_dim, rel_dim, n_layers):
        super(GNN, self).__init__()

        self.device = device

        self.source_node = source_node
        self.target_node = target_node
        self.use_matching_loss = use_matching_loss
        self.matching_hops = matching_hops

        self.types = types
        self.relations = relations
        self.n_layers = n_layers

        self.hid_dims = {}
        self.rel_dims = {}
        for type in types:
            self.hid_dims[type] = hid_dim
        for rel in relations:
            self.rel_dims[rel] = rel_dim

        # Matching loss
        self.matching_paths = metapath(self.types, self.relations, self.source_node, self.target_node, self.matching_hops)
        matching_w = {}
        for matching_id, matching_path in enumerate(self.matching_paths):
            for rel in reversed(matching_path[1:]):
                matching_w[str(matching_id) + rel] = nn.Linear(self.hid_dims[target_node], self.hid_dims[source_node], bias=False)
        self.matching_w = nn.ModuleDict(matching_w)
        for matching_id in self.matching_w.keys():
            nn.init.xavier_uniform_(self.matching_w[matching_id].weight)
        self.matching_loss = nn.MSELoss()

    def get_parameters(self):
        return NotImplementedError

    def convolve(self, h, edges, layer):
        return NotImplementedError

    def forward(self, feature, edges):
        h = {}
        for t in feature.keys():
            h[t] = self.adapt_w[t](feature[t])
        #cmd_loss = cmd(h[self.source_node], h[self.target_node])
        for layer in range(self.n_layers):
            h = self.convolve(h, edges, layer)
            #cmd_loss = cmd_loss + cmd(h[self.source_node], h[self.target_node])
        cmd_loss = cmd(h[self.source_node], h[self.target_node])
        return h, cmd_loss

    # Only for synthetic graphs
    def theoretical_transform(self, h_T):
        M_st = self.message_w['st'].weight.T
        M_tt = self.message_w['tt'].weight.T
        W_t = self.transform_w['target'].weight.T
        T1 = torch.matmul(M_st, W_t[:self.rel_dims['st']])
        T2 = torch.matmul(M_tt, W_t[self.rel_dims['st']:])
        T = torch.cat([T1, T2], axis=0)

        M_ss = self.message_w['ss'].weight.T
        M_ts = self.message_w['ts'].weight.T
        W_s = self.transform_w['source'].weight.T
        S1 = torch.matmul(M_ss, W_s[:self.rel_dims['ss']])
        S2 = torch.matmul(M_ts, W_s[self.rel_dims['ss']:])
        S = torch.cat([S1, S2], axis=0)

        return torch.matmul(h_T, torch.matmul(torch.pinverse(T), S))

    def transform(self, h_T):
        if self.use_matching_loss == False:
            return h_T
        trans_h_T = torch.zeros_like(h_T)
        for matching_id, matching_path in enumerate(self.matching_paths):
            h_Z = h_T
            for rel in reversed(matching_path[1:]):
                h_Z = self.matching_w[str(matching_id) + rel](h_Z)
                #h_Z = F.relu(h_Z)
            trans_h_T = trans_h_T + h_Z/len(self.matching_paths)
        return trans_h_T

    def get_matching_loss(self, edges, h_S, h_T):
        total_loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        if self.use_matching_loss == False:
            return total_loss
        for matching_id, matching_path in enumerate(self.matching_paths):
            h_Z = h_T
            for rel in reversed(matching_path[1:]):
                rel_source = self.relations[rel]['source']
                rel_target = self.relations[rel]['target']
                h_Z = self.matching_w[str(matching_id) + rel](h_Z)
                #h_Z = F.relu(h_Z)
                if len(edges[rel_target][rel_source][rel]['source']) == 0:
                    return total_loss
                h_Z = torch.spmm(edges[rel_target][rel_source][rel]['adj'], h_Z)
            total_loss = total_loss + self.matching_loss(h_S, h_Z)
        return total_loss


class HGNN(GNN):
    def __init__(self, device, source_node, target_node, use_matching_loss, matching_hops, \
                 types, relations, in_dims, hid_dim, rel_dim, n_layers=4):
        super(HGNN, self).__init__(device, source_node, target_node, use_matching_loss, matching_hops, \
                 types, relations, in_dims, hid_dim, rel_dim, n_layers)

        adapt_w = {}
        message_w = {}
        transform_w = {}
        for target_t in types.keys():
            adapt_w[target_t] = nn.Linear(in_dims[target_t], self.hid_dims[target_t], bias=False)
            for layer in range(n_layers):
                in_message_dim = self.hid_dims[target_t]
                for r in types[target_t]:
                    in_message_dim += self.rel_dims[r]
                    source_t = relations[r]['source']
                    message_w[r + str(layer)] = nn.Linear(self.hid_dims[source_t] + self.hid_dims[target_t], self.rel_dims[r], bias=False)
                transform_w[target_t + str(layer)] = nn.Linear(in_message_dim, self.hid_dims[target_t], bias=False)
        self.adapt_w = nn.ModuleDict(adapt_w)
        self.message_w = nn.ModuleDict(message_w)
        self.transform_w = nn.ModuleDict(transform_w)

        for target_t in self.types.keys():
            nn.init.xavier_uniform_(self.adapt_w[target_t].weight)
            for layer in range(n_layers):
                nn.init.xavier_uniform_(self.transform_w[target_t + str(layer)].weight)
                for r in self.types[target_t]:
                    nn.init.xavier_uniform_(self.message_w[r + str(layer)].weight)

    def get_parameters(self):
        ml = list()
        #ml.append({'params': self.minji_w.parameters()})
        for target_t in self.types.keys():
            ml.append({'params': self.adapt_w[target_t].parameters()})
            for layer in range(self.n_layers):
                ml.append({'params': self.transform_w[target_t + str(layer)].parameters()})
                for r in self.types[target_t]:
                    ml.append({'params': self.message_w[r + str(layer)].parameters()})
        for matching_id in self.matching_w.keys():
            ml.append({'params': self.matching_w[matching_id].parameters()})
        return ml

    def convolve(self, h, edges, layer):
        pooled_messages = defaultdict(list)
        for target_t in self.types.keys():
            for r in self.types[target_t]:
                source_t = self.relations[r]['source']
                if target_t not in edges.keys() or source_t not in edges[target_t].keys():
                    zero_message = torch.zeros((h[target_t].shape[0], self.rel_dims[r])).to(self.device)
                    pooled_messages[target_t].append(zero_message)
                    continue
                source_ids =  edges[target_t][source_t][r]['source']
                target_ids =  edges[target_t][source_t][r]['target']
                if len(source_ids) == 0:
                    zero_message = torch.zeros((h[target_t].shape[0], self.rel_dims[r])).to(self.device)
                    pooled_messages[target_t].append(zero_message)
                    continue
                source_h = h[source_t][source_ids]
                target_h = h[target_t][target_ids]
                message = self.message_w[r + str(layer)](torch.cat((source_h, target_h), axis=1))
                pooled_message = torch.spmm(edges[target_t][source_t][r]['agg_adj'], message).to(self.device)
                pooled_messages[target_t].append(pooled_message)

        for target_t in pooled_messages.keys():
            h_old = h[target_t]
            pooled_message = pooled_messages[target_t]
            pooled_message.append(h_old)
            h_new = self.transform_w[target_t + str(layer)](torch.cat(pooled_message, axis=1))
            h_new = h_new + h_old
            h_new = F.relu(h_new)
            h[target_t] = h_new
        return h
